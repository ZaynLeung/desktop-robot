#!/usr/bin/env node
// 校验一个 DSH skill bundle 是否符合发现与解析规则。
// 用法: node validate-skill.mjs <skill-directory | path/to/SKILL.md>
// 退出码: 0 通过（可能带警告），1 有错误，2 用法/路径错误。

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';

const NAME_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const MAX_SKILL_CHARS = 8192;
const MAX_DESCRIPTION = 500;
const RESOURCE_DIRS = new Set(['references', 'scripts', 'assets']);
const ALLOWED_KEYS = new Set([
  'name',
  'description',
  'whenToUse',
  'metadata',
  'user-invocable',
  'disable-model-invocation',
]);
const BOOL_KEYS = ['user-invocable', 'disable-model-invocation'];
const BOOL_VALUES = new Set(['true', 'false', 'yes', 'no', 'on', 'off', '1', '0']);

const errors = [];
const warnings = [];
const report = [];
const fail = (msg) => errors.push(msg);
const warn = (msg) => warnings.push(msg);

main();

function main() {
  const input = process.argv[2];
  if (!input) {
    console.error('用法: node validate-skill.mjs <skill-directory | path/to/SKILL.md>');
    process.exit(2);
  }

  const target = resolve(input);
  if (!existsSync(target)) {
    console.error(`路径不存在: ${target}`);
    process.exit(2);
  }

  const isDir = statSync(target).isDirectory();
  const skillDir = isDir ? target : dirname(target);
  const skillFile = isDir ? join(target, 'SKILL.md') : target;
  const dirName = basename(skillDir);

  report.push(`skill 目录: ${skillDir}`);

  if (basename(skillFile) !== 'SKILL.md') {
    fail(`入口文件名必须是 SKILL.md，实际是 ${basename(skillFile)}`);
    finish();
    return;
  }
  if (!existsSync(skillFile) || !statSync(skillFile).isFile()) {
    fail(`找不到 ${skillFile}（目录 bundle 必须是 <name>/SKILL.md）`);
    finish();
    return;
  }
  if (!NAME_RE.test(dirName)) {
    fail(`目录名不是 kebab-case: "${dirName}"`);
  }

  const raw = readFileSync(skillFile, 'utf8');
  const text = raw.replace(/^\uFEFF/, '');
  const chars = [...text].length;
  const bytes = Buffer.byteLength(text, 'utf8');
  const lines = text.split(/\r?\n/);

  if (lines[0]?.trim() !== '---') {
    fail('第一行必须是 "---"（YAML frontmatter 起始）');
    finish();
    return;
  }
  let end = -1;
  for (let i = 1; i < lines.length; i += 1) {
    if (lines[i].trim() === '---') {
      end = i;
      break;
    }
  }
  if (end < 0) {
    fail('缺少 frontmatter 结束的 "---"');
    finish();
    return;
  }

  const fm = parseFrontmatter(lines.slice(1, end));
  const body = lines.slice(end + 1).join('\n').trim();

  if (!('name' in fm)) fail('frontmatter 缺少必填字段 name');
  if (!('description' in fm)) fail('frontmatter 缺少必填字段 description');

  if (fm.name !== undefined) {
    if (!NAME_RE.test(fm.name)) {
      fail(`name 不是 kebab-case: "${fm.name}"（只允许小写字母、数字、单个连字符）`);
    } else if (fm.name !== dirName) {
      warn(`name "${fm.name}" 与目录名 "${dirName}" 不一致，预览和排障会对不上`);
    }
  }
  if (fm.description !== undefined) {
    if (fm.description.length === 0) fail('description 为空');
    if (fm.description.length > MAX_DESCRIPTION) {
      warn(`description ${fm.description.length} 字符，超过会话目录上限 ${MAX_DESCRIPTION}，会被截断`);
    }
  }
  for (const key of Object.keys(fm)) {
    if (!ALLOWED_KEYS.has(key)) warn(`未知 frontmatter 字段 "${key}"，可能不会生效`);
  }
  for (const key of BOOL_KEYS) {
    if (fm[key] === undefined) continue;
    if (!BOOL_VALUES.has(String(fm[key]).toLowerCase())) {
      fail(`${key} 的值 "${fm[key]}" 不是严格布尔（true/false/yes/no/on/off/1/0），整个 skill 会被丢弃`);
    }
  }

  if (!body) fail('frontmatter 之后没有正文');
  if (chars > MAX_SKILL_CHARS) {
    warn(`SKILL.md 为 ${chars} 字符，超过 ${MAX_SKILL_CHARS}，可能被工具结果修剪器截断；把细节拆进 references/`);
  }
  report.push(`SKILL.md: ${chars} 字符 / ${bytes} 字节`);

  for (const nested of findNestedSkillFiles(skillDir)) {
    fail(`嵌套的 ${relative(skillDir, nested)} 不会被发现：bundle 必须是扫描根的直属子目录`);
  }

  const referenced = collectReferences(body);
  for (const ref of referenced) {
    if (ref.includes('..') || ref.startsWith('/') || /^[A-Za-z]:/.test(ref)) {
      warn(`引用了绝对路径或越界路径 "${ref}"，应使用相对 skill 目录的路径`);
      continue;
    }
    if (!existsSync(join(skillDir, ref))) {
      fail(`正文引用了不存在的文件 "${ref}"`);
    }
  }
  report.push(`正文引用资源: ${referenced.size} 个`);

  finish();
}

function finish() {
  for (const line of report) console.log(`INFO  ${line}`);
  for (const msg of warnings) console.log(`WARN  ${msg}`);
  for (const msg of errors) console.log(`ERROR ${msg}`);
  if (errors.length > 0) {
    console.log(`\n结果: 失败（${errors.length} 个错误，${warnings.length} 个警告）`);
    process.exitCode = 1;
  } else {
    console.log(`\n结果: 通过（${warnings.length} 个警告）`);
    process.exitCode = 0;
  }
}

function parseFrontmatter(sourceLines) {
  const result = {};
  let pendingKey = null;
  let pendingValue = [];
  const flush = () => {
    if (pendingKey) result[pendingKey] = pendingValue.join(' ').trim();
    pendingKey = null;
    pendingValue = [];
  };

  for (const line of sourceLines) {
    if (line.trim() === '' || /^\s*#/.test(line)) continue;
    if (/^\s+\S/.test(line)) {
      if (pendingKey) pendingValue.push(line.trim());
      // 缩进行是映射值（例如 metadata 的子键），不参与顶层校验。
      continue;
    }
    flush();
    const match = line.match(/^([A-Za-z0-9_-]+):(?:\s*(.*))?$/);
    if (!match) {
      warn(`无法解析 frontmatter 行: ${line}`);
      continue;
    }
    const key = match[1];
    const value = (match[2] ?? '').trim();
    if (['|', '>', '|-', '>-', '|+', '>+'].includes(value)) {
      pendingKey = key;
      pendingValue = [];
      continue;
    }
    result[key] = stripQuotes(value);
  }
  flush();
  return result;
}

function stripQuotes(value) {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function collectReferences(body) {
  const found = new Set();
  for (const match of body.matchAll(/`([^`\n]+)`/g)) {
    const token = match[1].trim();
    if (/^(references|scripts|assets)\/\S+/.test(token)) found.add(token);
  }
  for (const match of body.matchAll(/\]\(([^)\s]+)\)/g)) {
    const token = match[1].trim();
    if (!/^(https?:|#|mailto:)/.test(token)) found.add(token.replace(/^\.\//, ''));
  }
  return found;
}

function findNestedSkillFiles(root) {
  const found = [];
  const walk = (dir, depth) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      if (entry.name === 'node_modules' || entry.name === '.git') continue;
      if (depth === 0 && RESOURCE_DIRS.has(entry.name)) continue;
      const full = join(dir, entry.name);
      try {
        if (statSync(join(full, 'SKILL.md')).isFile()) found.push(join(full, 'SKILL.md'));
      } catch {
        /* 没有 SKILL.md */
      }
      walk(full, depth + 1);
    }
  };
  walk(root, 0);
  return found;
}
