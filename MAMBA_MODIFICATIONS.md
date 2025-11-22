# Mamba-Subgoal 项目的 RecBole 修改记录

这个文件记录了为SubgoalSlateRL项目对RecBole所做的所有修改。

## 分支信息
- **分支名称**: mamba-subgoal-modifications
- **基于版本**: RecBole v1.2.1
- **创建日期**: 2025-10-31

## 修改计划

### Phase 1: 基础模型支持
- [ ] 添加Mamba模型实现
- [ ] 添加Transformer Slate解码器
- [ ] 支持序列到列表的输出格式

### Phase 2: 训练支持
- [ ] 集成IQL离线强化学习
- [ ] 添加子目标对齐损失
- [ ] 支持多目标约束优化

### Phase 3: 评估指标
- [ ] 多样性指标
- [ ] 覆盖率指标
- [ ] 位置偏差感知指标

## 修改日志

| 日期 | 文件 | 修改内容 | 状态 |
|------|------|----------|------|
| 2025-10-31 | MAMBA_MODIFICATIONS.md | 创建修改记录文件 | ✅ |
| 2025-10-31 | mambasubgoal.py, dataset.py, __init__.py | 注册MambaSubgoal为RecBole序列推荐模型 | ✅ |
| 2025-11-22 | trainer.py | 修复PyTorch 2.6+兼容性问题 (weights_only=False) | ✅ |

---
**注意**: 此分支的所有修改仅用于mamba_subgoal项目，不会影响RecBole官方代码。
