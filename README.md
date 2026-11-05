# desktop

## Skills

本项目自带一个用于生成 skill 的 skill：[skill-creator](.agents/skills/skill-creator/SKILL.md)。

- 调用：在对话里说「帮我生成一个 skill」，或输入 `/skill-creator`。
- 存放：新 skill 写到 `.agents/skills/<name>/SKILL.md`（DSH 只扫描根目录下一层，嵌套目录不会被发现）。
- 校验：`node .agents/skills/skill-creator/scripts/validate-skill.mjs <skill-directory>`。
- 规范与模板：`.agents/skills/skill-creator/references/`、`.agents/skills/skill-creator/assets/SKILL.template.md`。

> 若运行上面的 node 命令时报沙箱授权失败（Windows 文件权限问题），把会话切到「完全权限」后再运行。
