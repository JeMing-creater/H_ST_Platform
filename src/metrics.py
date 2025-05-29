import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score
from sklearn.metrics import roc_auc_score
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc

def compute_sp_metrics(risks, dd_labels, vs_labels, accelerator):
    # 使用 accelerator.gather_for_metrics 收集所有进程的数据
    risks = torch.cat(risks)
    dd_labels = torch.cat(dd_labels)
    vs_labels = torch.cat(vs_labels)

    risks, dd_labels, vs_labels = accelerator.gather_for_metrics((risks, dd_labels, vs_labels))

    # 在主进程上计算评估指标
    if accelerator.is_main_process:
        risks = risks.cpu().detach().numpy().flatten()
        dd_labels = dd_labels.cpu().detach().numpy().flatten()
        vs_labels = vs_labels.cpu().detach().numpy().flatten()

        # 计算一致性指数（CI）
        ci = concordance_index(dd_labels, -risks, vs_labels)

        # 构造生存数据结构
        y_true = Surv.from_arrays(vs_labels == 1, dd_labels)

        # 设置评估的时间点，确保在测试数据的随访时间范围内
        min_time = np.min(dd_labels)
        max_time = np.max(dd_labels)
        times = np.linspace(min_time, max_time * 0.999, 100)

        # 估计生存概率，假设风险值为对数风险，使用指数函数转换为生存概率
        surv_probs = np.exp(-np.outer(risks, times))

        # 计算Brier分数（BS）
        _, bs_scores = brier_score(y_true, y_true, surv_probs, times)
        bs_mean = np.mean(bs_scores)

        # 计算 AUC
        y_val = Surv.from_arrays(event=vs_labels == 1, time=dd_labels)
        try:
            _, auc = cumulative_dynamic_auc(y_val, y_val, -risks, times)
        except ValueError:
            print('vs labels all 0 or 1')
            auc = 0.0

        return ci, bs_mean, auc
    else:
        return None, None, None


def km_analysis(risks, dd_labels, vs_labels, accelerator, save_path=None):
    risks = torch.cat(risks)
    dd_labels = torch.cat(dd_labels)
    vs_labels = torch.cat(vs_labels)

    # gather across processes (multi-GPU)
    risks, dd_labels, vs_labels = accelerator.gather_for_metrics((risks, dd_labels, vs_labels))

    # only main process plots
    if accelerator.is_main_process:
        # 转为 numpy 并 flatten
        risks = risks.cpu().detach().numpy().flatten()
        dd_labels = dd_labels.cpu().detach().numpy().flatten()
        vs_labels = vs_labels.cpu().detach().numpy().flatten()

        # accelerator.print("Risk Score Range:", risks.min(), risks.max(), "Median:", np.median(risks))
        # accelerator.print("Number High Risk:", np.sum(risks > np.median(risks)))
        # accelerator.print("Number Low Risk:", np.sum(risks <= np.median(risks)))


        df = pd.DataFrame({
            "time": dd_labels,
            "event": vs_labels,
            "risk": risks
        })

        median_risk = df["risk"].median()
        df["group"] = np.where(df["risk"] >= median_risk, "High Risk", "Low Risk")

        
        # sorted_idx = np.argsort(df["risk"].values)
        # n = len(df)
        # df["group"] = "Low Risk"
        # df.loc[sorted_idx[n//2:], "group"] = "High Risk"
        
        plt.figure(figsize=(10, 6))
        for name, group_df in df.groupby("group"):
            kmf = KaplanMeierFitter()  # ✅ 每组一个新实例，避免状态干扰
            kmf.fit(durations=group_df["time"], event_observed=group_df["event"], label=name)
            kmf.plot_survival_function()

        plt.title("Kaplan-Meier Survival Curve by Model Risk")
        plt.xlabel("Normalized Time (0~1)")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name != "" and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            if os.path.exists(save_path):
                os.remove(save_path)
            plt.savefig(save_path)
            print(f"Survival curve saved to: {save_path}")

        plt.show()