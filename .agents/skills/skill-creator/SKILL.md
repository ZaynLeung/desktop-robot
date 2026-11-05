---
name: skill-creator
description: 把一类重复任务沉淀为可复用的 skill：先定边界与触发描述，再创建目录 bundle 与 SKILL.md，编写按需加载的 references/scripts/assets，最后按 DSH 规范校验可发现性与加载效果。用户说要生成、编写、修改或检查 skill 时使用。
whenToUse: 用户要求生成或修改 skill（技能），或需要把某套流程、领域知识、脚本用法固化成可复用技能时。
---

# 生成 Skill

目标：产出一个**能被发现、能按需加载、能被真正执行**的 skill 目录，而不是一段散文说明。按顺序执行，不要跳过第 6、7 步。

## 1. 先定边界，先写 description

- 一个 skill 只服务一类任务。需求跨了两类任务（例如「写周报」和「发布上线」）就拆成两个；只做入口分流时写一个 router skill。
- `description` 是模型唯一的路由依据：写成「做什么 + 什么时候用 + 用户可能的说法」。一句话，≤500 字符，禁止「一个强大的技能」这类空话。
- 已有同类 skill 时优先改它。同名或近义词 skill 会让路由摇摆，也会让旧行为残留。

## 2. 选存放位置

skill 只从扫描根**一层**内发现，嵌套的 `**/SKILL.md` 不会被识别。

| 位置 | 来源 | 适用 |
| --- | --- | --- |
| `<项目>/.agents/skills/` | project-agents | 跨 agent 共享的项目 skill，默认选它 |
| `<项目>/.dsh/skills/` | project-dsh | 只给本项目 DSH 用 |
| `~/.agents/skills/`、`~/.dsh/skills/` | user-* | 本机所有项目可用 |
| 配置项 `customSkillDirs` | custom | 自定义根目录 |

形态只有两种：目录 bundle `<name>/SKILL.md`（带资源时用它），或平铺文件 `<name>.md`（单文件够用时）。目录名与 frontmatter 的 `name` 保持一致。

## 3. 搭骨架

```text
<name>/
├── SKILL.md            # 必需：frontmatter + 正文
├── references/         # 长清单、格式规范、示例；正文里说明何时读
├── scripts/            # 确定性的活（校验、转换、生成）写成脚本
└── assets/             # 模板、样例文件
```

只建真正会用到的子目录。每个资源文件都必须在正文里有明确的读取入口，否则等于不存在。

## 4. 写 frontmatter

```yaml
---
name: lower-kebab-case-name
description: 一句话，说明做什么、何时用、关键说法。
whenToUse: 可选，补充触发场景。
metadata:              # 可选，自由键值
  owner: example
user-invocable: true               # 可选，默认 true；false 则不出现在 / 菜单
disable-model-invocation: false    # 可选，默认 false；true 则模型不可见，只能 /name 调用
---
```

硬性规则：`name` 必须 kebab-case；`description` 必填；两个调用布尔键只接受严格布尔（`true/false`、`yes/no`、`on/off`、`1/0`），写错会让整个 skill 被静默丢弃。字段细节与 YAML 陷阱见 `references/frontmatter.md`。

## 5. 写正文

- 用祈使句写**动作**，不写动机和背景故事；每条指令都要能直接执行。
- 按「何时用 → 步骤 → 失败处理 → 验证」组织；步骤编号，一步一件事。
- 正文只放每次都要用的内容。能按需读的细节移到 references 目录，并在需要的位置写明「遇到 X 时读 references 下的哪一份」。
- 引用资源用相对路径（例如 `references/frontmatter.md`），运行时会按 skill 目录解析。
- 单个 SKILL.md 控制在 8192 字符以内（约 8 KB），超过会被工具结果修剪器截断；超出就拆进 references。
- 给可复制的最小示例，好例与坏例对照，比抽象描述有效。

四种常见结构（流程型、路由型、规范型、工具封装型）和选型建议见 `references/structure-patterns.md`；也可以直接复制 `assets/SKILL.template.md`。

## 6. 校验

逐项过 `references/validation-checklist.md`，至少确认：目录直接位于扫描根下、`name` 合法且与目录同名、description 含触发词、正文引用的每个文件真实存在、没有嵌套 bundle、没有同名冲突。

有 node 时运行：

```bash
node scripts/validate-skill.mjs <skill-directory>
```

## 7. 验证加载

用 `skill` 工具按 `name` 加载刚写的 skill：能返回正文，说明它已被发现、可解析、可加载。加载失败按 checklist 的失败模式排查，通常是目录层级、frontmatter 语法或 name 不合法。加载成功后，把返回内容与需求逐条对照，不要只确认文件存在。
