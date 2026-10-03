"""Create an honest poster-development report and exportable figure from T03 tables."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'results/t03'
LABELS={'ZERO':'Zero forecast','PERSISTENCE':'Persistence','DAMPED':'Damped persistence','HOST':'AsymODE host','DOSE':'AsymODE + weather memory','GCRK':'AsymODE + weather memory + GCRK','GEO_MLP':'AsymODE + weather memory + geo MLP','TREE_W':'Weather tree','TREE_G':'Weather + geography tree','BLEND':'Validation-selected ensemble'}

def main():
    m=pd.read_csv(OUT/'metrics.csv');c=pd.read_csv(OUT/'comparisons.csv');events=pd.read_csv(OUT/'per_event.csv')
    full=m[(m.weighting=='design')&(m.scope=='full144')&(m.group=='all')].set_index('model')
    exact=m[(m.weighting=='design')&(m.scope!='full144')&(m.group=='all')]
    comparison=c.set_index(['model','reference']);blend=comparison.loc[('BLEND','HOST')];kernel=comparison.loc[('GCRK','DOSE')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':18,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'pdf.fonttype':42,'ps.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(16,7.2),gridspec_kw={'width_ratios':[1.2,1]})
    display=['HOST','DOSE','GCRK','GEO_MLP','TREE_W','TREE_G','BLEND']
    short=['AsymODE host','+ weather memory','+ memory + GCRK','+ memory + geo MLP','Weather tree','Weather + geo tree','Ensemble']
    colors=['#96a8b1','#709da9','#c27572','#caa493','#a8bfcd','#799ba8','#315c70']
    vals=full.loc[display,'rmse'].to_numpy();yy=np.arange(len(display))
    ax.barh(yy,vals,color=colors,height=.68);ax.set_yticks(yy,short);ax.invert_yaxis()
    for y,v in zip(yy,vals):ax.text(v+vals.max()*.016,y,f'{v:.5f}',va='center',fontsize=13)
    ax.set_xlim(0,vals.max()*1.27);ax.set_xlabel('Pooled OSI RMSE, full 144-hour path')
    ax.set_title('(a) Complete trajectory accuracy',loc='left',pad=18)
    horizons=[1,6,24,48]
    for arm,col,marker in [('HOST','#96a8b1','o'),('GCRK','#c27572','s'),('BLEND','#315c70','D')]:
        line=[float(exact[(exact.model==arm)&(exact.scope==f'exact+{h}')].iloc[0].rmse) for h in horizons]
        bx.plot(range(4),line,color=col,marker=marker,lw=2.5,ms=7,label={'HOST':'AsymODE host','GCRK':'+ weather memory + GCRK','BLEND':'Ensemble'}[arm])
    bx.set_xticks(range(4),['+1 h','+6 h','+24 h','+48 h']);bx.set_xlabel('Exact lead from one fixed origin');bx.set_ylabel('Pooled OSI RMSE')
    bx.set_ylim(bottom=0);bx.set_title('(b) Fixed-origin forecast leads',loc='left',pad=18);bx.legend(frameon=False,fontsize=12,loc='best')
    for a in (ax,bx):
        a.spines[['top','right']].set_visible(False);a.grid(axis='x' if a is ax else 'y',color='#e8ecee',lw=.8);a.set_axisbelow(True)
    fig.suptitle('Public tropical outage data: county-held-out prediction',fontsize=22,x=.53,y=.97)
    fig.text(.5,.065,'15 tropical event windows | 774 counties | 1,633 county-events | 3 neural seeds | 5 county folds',ha='center',fontsize=13)
    fig.text(.5,.027,'Conditional hindcast with supplied weather; examined development data. Ensemble gains do not establish a kernel-specific benefit.',ha='center',fontsize=11,color='#525b61')
    fig.subplots_adjust(left=.20,right=.98,top=.85,bottom=.20,wspace=.43)
    fig.savefig(OUT/'poster_trajectory_results.pdf');fig.savefig(OUT/'poster_trajectory_results.png',dpi=220);plt.close(fig)
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{County-held-out accuracy on 15 public tropical events. Metrics are design-weighted and pooled over valid observations. Four leads are exact times from a fixed origin. This is a conditional hindcast with supplied weather on examined development data.}',r'\label{tab:public-tropical-t03}',r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'\multirow{2}{*}{Model} & \multicolumn{4}{c}{MAE} & \multicolumn{4}{c}{RMSE} \\',r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',r' & $+1$ h & $+6$ h & $+24$ h & $+48$ h & $+1$ h & $+6$ h & $+24$ h & $+48$ h \\',r'\midrule']
    for arm in full.index:
        values=[]
        for measure in ('mae','rmse'):
            values.extend(f'{float(exact[(exact.model==arm)&(exact.scope==f"exact+{h}")].iloc[0][measure]):.5f}' for h in horizons)
        tex.append(LABELS[arm]+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    (OUT/'poster_table.tex').write_text('\n'.join(tex))
    lines=['# T03：公开热带停电数据完整轨迹结果', '', '2026-10-03。全部15个热带开发事件，774县、1,633个县事件；全局县级五折，同县跨事件始终同折。训练seeds 0/1/2，每个神经臂15个拟合，共60个神经拟合。树模型5折×2臂×3配置，共30个拟合。',
        '', '## 可以展示的结果', '', f'验证集选权重的集成相对HOST，完整144小时加权pooled RMSE改善 **{blend.rmse_gain_pct:.2f}%**；县簇重采样95%区间 **[{blend.ci025_pct:.2f}%, {blend.ci975_pct:.2f}%]**。该区间条件于既有模型和划分，未包含重训练或数据选择不确定性，多候选比较未作多重校正。',
        '', '|模型|完整路径MAE|完整路径RMSE|相对HOST改善|', '|---|---:|---:|---:|']
    for arm,r in full.iterrows():
        gain=100*(1-r.rmse/full.loc['HOST','rmse']);lines.append(f'|{LABELS[arm]}|{r.mae:.5f}|{r.rmse:.5f}|{gain:+.2f}%|')
    unweighted=m[(m.weighting=='unweighted')&(m.scope=='full144')&(m.group=='all')].set_index('model')
    ug=100*(1-unweighted.loc['BLEND','rmse']/unweighted.loc['HOST','rmse'])
    mg=100*(1-full.loc['BLEND','mae']/full.loc['HOST','mae'])
    lines += ['', f'集成完整路径MAE改善{mg:.2f}%；不加设计权重时，HOST/集成RMSE为{unweighted.loc["HOST","rmse"]:.5f}/{unweighted.loc["BLEND","rmse"]:.5f}，改善{ug:.2f}%。']
    lines += ['', '## 地理核是否有独立贡献', '', '|比较|RMSE改善|县簇95%区间|', '|---|---:|---:|']
    for base in ('HOST','DOSE','GEO_MLP'):
        r=comparison.loc[('GCRK',base)];lines.append(f'|GCRK 对 {base}|{r.rmse_gain_pct:+.2f}%|[{r.ci025_pct:+.2f}%, {r.ci975_pct:+.2f}%]|')
    if kernel.ci025_pct>0:
        lines.append('\n本轮GCRK相对相同天气记忆输入的DOSE有正向区间证据，但仍需结合直接地理MLP对照和已查看开发集的范围解释。')
    else:lines.append('\n本轮不能宣称地理核有稳定独立增益。整体集成的改善应归于实际模型组合；不能全部归功于GCRK。')
    wins=events.pivot(index='system',columns='model',values='rmse')
    lines += ['', f'集成相对HOST在{int((wins.BLEND<wins.HOST).sum())}/15个事件的完整路径RMSE改善；逐事件完整正负结果见`per_event.csv`。',
        '', '## 训练与信息边界', '', '- 输入为历史72小时停电摘要和天气序列；从最后观测OSI出发预测未来144小时，不读取未来停电。HOST恢复率逐小时更新。DOSE加入6/24/72小时均值、从窗口开始的最大值和0.9/0.98指数记忆。',
        '- GCRK使用仓库原始4维地理编码的地理条件化隐藏响应核，接收geo40；GEO_MLP在第一损伤层加入零初始化地理线性投影。不是I20 controlled-relaxation新核；二者容量不同，属于架构对照。',
        '- 四个神经臂按同一seed配对初始化，各训练600步；Adam主干0.003、恢复0.0003，batch最多64，梯度范数1。所有参数可更新；每50步按内层县验证MSE选checkpoint，包含第0步，无外层选点。',
        '- 地理核初次前向和每10步仅在fit县校准；保留其原始warmup和drop-path。历史摘要为last/mean/max/coverage，base输入58维，DOSE输入106维。',
        '- 树模型用同样天气/历史/记忆，加预测lead；预测相对训练侧选择的衰减持续值的残差。三组固定参数由内层县验证选择；早停50轮、最多500轮。所有评分和选择均在裁剪至[0,1]后计算。',
        '- 神经臂先平均三seed预测；再用内层验证选六个候选的非负、和为1的权重。每折一组权重用于整个144小时，不按时效选模型。',
        '- 所有事件和原观测mask均保留。输入填补/标准化仅fit县；树保留缺失标记；缺失HRRR小时设为NaN并加availability，不把缓存内填补值当实测。',
        '- ERA5以及未来小时HRRR f01来自当时预测起点之后的信息；这是给定天气条件的hindcast，不能声称实时六天预报。静态地理数据也未断言在历史事件时均已可得。',
        '- D已在此前研究中查看，不是独立确认测试；同事件有训练与留出县，不要求跨事件泛化。',
        '- 停电标签来自EAGLE-I，分母沿用现有面板的2024建模客户数；历史事件并非都使用当年实测客户总数。15个窗口按现有目录归类，部分未匹配best track，不宣称15场独立飓风。完整数据说明见`data_v1/README.md`。',
        '', '## 评估和核验', '', '- 主指标：设计权重的全144小时pooled MAE/RMSE。附未加权结果、严重/非严重子组、逐seed及逐事件。',
        '- 四时效表是固定起点的精确+1/+6/+24/+48小时（索引0/5/23/47），不是比赛中从该时效起算的后缀切片，不能直接混表比较。',
        '- 代码检查全部县OOF恰好一次、内外县不重叠、预测有限且在[0,1]、配对初始化一致；由每折SSE/SAE与权重重建pooled指标。验证结果见`verification.json`。',
        '- 中间未压缩缓存首次不完整，改压缩并增加CRC及SHA核验后重启；原GCRK任务在首次前向缺少校准时停止，补上fit-only校准后补齐。没有根据外层分数改变候选或训练预算。',
        '', '## Poster使用建议', '', '可以展示公开数据、县留出、完整预测路径及实际模型的误差改善。标题和图注必须保留“conditional hindcast / supplied weather / development evaluation”的信息边界。地理核贡献只能引用上面的匹配对照。',
        '', '数据和现有窗口继续保留；下一步先核验真正预报天气的起报时间，再把较好的训练/集成设计放到明确的确认数据上。若需要继续改核，使用新的开发阶段并完整记录候选，不把本轮已查看结果重新称作未见测试。',
        '', '交付：`poster_trajectory_results.pdf/png`、`poster_table.tex`、`metrics.csv`、`comparisons.csv`、`per_event.csv`、`per_seed.csv`，以及代码、协议、参数选择和核验记录。逐县预测及权重仅本地保留。','']
    (HERE/'notes/T03_POSTER_TRAJECTORY_RESULTS_20261003_ZH.md').write_text('\n'.join(lines))
    print('report and poster figure written')

if __name__=='__main__':main()
