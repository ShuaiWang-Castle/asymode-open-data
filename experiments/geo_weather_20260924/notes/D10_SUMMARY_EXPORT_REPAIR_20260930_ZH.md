# D10纯汇总导出修复登记（2026-09-30）

原D10源码登记1ca63c44cc18c42397bdcd777ca44f013fa7db32保持不变。两个worker已退出，
各自完成W/new全留出和共同FIT重放及四个干预、全部117/183严重病例的双臂oracle，
8份完整数值缓存已保存。最终json.dump没有使用strict_json，遇到np.int64停止。
保留每折FAILED.json及部分RESULT.json和所有NPZ，不覆盖，不再前向/重建/训练/校准模型。

新d10_cache_recover.py仅在本登记Git blob及原源/六DONE/output守卫完成后读既有缓存，
重建原汇总并使用strict_json导出到全新v2文件。首次oracle的逐例证书未写入缓存而随
内存退出丢失；故仅从保存的y/m/y0/r/gate/background重复完全相同600个凸QP以重建证书，
无新oracle、参数网格、病例或模型计算。核对既有可行path的约束/原目标/重构，给出
重建path差异与旧path上界；不把数值复算叫作额外独立试验或新模型成绩。

所有首次缓存和失败/部分文件在读前及结束后核对下列哈希；范围、指标、干预、容差和
病例保留规则不变。最多两进程各两线程nice15，旧监控继续暂停。

```json
{
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/FAILED.json": "523fd0eb06248dd23f7209637e4115e6f00aba6edd0207e8445b64afaba67a34",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/RESULT.json": "d5b5cc6ed3b9fb42f3b59096503f7428c9e082a61156ea02c0c3815838481101",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/host_common_FIT.npz": "9c531ac59f5c3a5f0a45e42fc068523db4b92b13bda442a8c730daabfa69c0da",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/host_full_heldout.npz": "9b524b2f6ba3cf0e29e15574d76abdd7127e5170054747b0500f0f2d26eda4d6",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/new_common_FIT.npz": "7a467fcd9e1f3c14bd61ef0e72378c3c7739d9aaf809409f85734d8881735aa4",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold2/new_full_heldout.npz": "9fce97d0e2566a2acff70ef09e75ab4328a49287de23059835bd6a4f4547b22b",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/FAILED.json": "523fd0eb06248dd23f7209637e4115e6f00aba6edd0207e8445b64afaba67a34",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/RESULT.json": "a944d61f93331f8a5f815aa929b4bf750d69f9c64a5f4485a4367432207509e0",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/host_common_FIT.npz": "f03c51d5001dd3a8ec64ada67d5c88db602893266a2a88acf9124ad3b369097f",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/host_full_heldout.npz": "1ede1b0425c3c21160669e71cc58558a9a951af995cb5205d0bdcda57fd74831",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/new_common_FIT.npz": "a81c27397055ce30fcfd700b9cabc40e3372f883ec40a92df9819ab37698061a",
  "runs/geo_weather_20260924/d10_response_chain_20260929/fold3/new_full_heldout.npz": "7df17e0783c20b52a06b5d44246ce8f662249201a8f945765f0b9d62d8633f69"
}
```
