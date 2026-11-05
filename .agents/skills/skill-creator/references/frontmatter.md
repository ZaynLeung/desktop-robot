# SKILL.md frontmatter 规范

写坏了不会被报错，只会被**静默跳过**：目录在、文件在，但 skill 不出现。下面每一条都对应一次真实的失败模式。

## 发现模型

- 扫描根只扫一层：`<root>/<name>/SKILL.md`（目录 bundle）或 `<root>/<name>.md`（平铺文件）。
- `<root>/<name>/nested/SKILL.md` 不会被发现。bundle 必须是扫描根的直属子目录。
- 多个来源会合并，同名时按优先级取一个：

| 优先级 | 来源 | 路径 |
| --- | --- | --- |
| 1 | project-dsh | `<项目>/.dsh/skills` |
| 2 | project-agents | `<项目>/.agents/skills` |
| 3 | custom | 配置项 `customSkillDirs` |
| 4 | user-dsh | `~/.dsh/skills` |
| 5 | user-agents | `~/.agents/skills` |

- 文件必须以 YAML frontmatter 开头（第一行 `---`），正文跟在结束的 `---` 之后。
- 会话目录里的 description 会被截断（默认上限 500 字符），所以最重要的触发词放最前面。

## 字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | kebab-case 唯一标识，也是 `skill` 工具的加载名和用户 `/name` 的调用名 |
| `description` | 是 | 路由描述，模型只凭它决定是否加载 |
| `whenToUse` | 否 | 补充触发与排除条件，适合写「在 X 时用，Y 时不用」 |
| `metadata` | 否 | 自由键值映射（版本、负责人、来源），不参与路由 |
| `disable-model-invocation` | 否 | 默认 `false`。`true` 时模型看不到也不能加载，只能由用户 `/name` 调用 |
| `user-invocable` | 否 | 默认 `true`。`false` 时不出现在 `/` 菜单，模型仍可加载 |

最小可加载示例：

```yaml
---
name: release-notes
description: 从 git 提交生成发布说明。用户要求写 changelog、release notes 或发版说明时使用。
---
```

## 严格布尔

两个调用键只接受：`true`/`false`、`yes`/`no`、`on`/`off`、`1`/`0`（不区分大小写）。
`True` 可以；`ture`、`"true"`（加引号会变成字符串，别加）、`enabled` 会让整个 skill 被丢弃。省略即默认允许。

## name 规则

- 只允许 ASCII 小写字母、数字和单个连字符：`release-notes`、`office-xlsx`、`s3-sync`。
- 不能有空格、下划线、大写、连续连字符，不能以连字符开头或结尾。
- 目录名、平铺文件名、frontmatter 的 `name` 三者保持一致，否则预览、路由和排障都会对不上。

## description 规则

一句话，包含三件事：做什么、什么时候用、用户会怎么说。

好：

```yaml
description: 把 Word 文档转成 PDF 并检查版式。用户要求导出 PDF、打印预览或检查分页时使用。
```

坏：

```yaml
description: 一个强大的文档处理技能。
```

坏在哪里：没有动作对象（Word→PDF）、没有触发词（导出 PDF、打印预览、分页），模型无法判断何时加载；「强大」不携带任何信息。

`whenToUse` 用来补排除条件，例如「只在需要写文件时用，只读查询用 `reports-reader`」。

## YAML 陷阱

- 值里出现英文冒号加空格（`: `）必须整体加引号，否则被当成嵌套键：`description: "修复 bug: 定位并提交补丁"`。中文冒号「：」没有这个问题。
- 以 `#` 开头的值会被当注释，需要引号。
- 多行 description 用块标量，并保证有内容：`description: >-` 之后缩进写正文。
- 不要用 tab 缩进；统一两个空格。
- 键名不要自创：除 `name`、`description`、`whenToUse`、`metadata` 外，调用相关的键只有 `user-invocable` 与 `disable-model-invocation`，写错名字不会生效。

## 刷新时机

- 通过 DSH 的 `write` / `edit` 写入会同步刷新会话目录，新 skill 立刻可加载。
- 用编辑器在磁盘上手改、增删 SKILL.md 不会主动通知已开启的会话；需要新建会话、切换 preset 或重连才会看到变化。排查「明明写了却不出现」时先排除这一点。
