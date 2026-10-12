'use strict';
// Entry point run by .github/workflows/review-bot.yml. Node 20, no dependencies.
const fs = require('fs'); const crypto = require('crypto');
const L = require('./lib');
const env = process.env;
const [owner, repo] = env.GITHUB_REPOSITORY.split('/');
const pr = Number(env.PR_NUMBER);
const gh = async (path, opts = {}) => {
  const r = await fetch(`https://api.github.com/repos/${owner}/${repo}${path}`, {
    ...opts, headers: { Authorization: `Bearer ${env.GH_TOKEN}`, Accept: opts.accept || 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', ...(opts.body ? { 'Content-Type': 'application/json' } : {}) },
  });
  if (!r.ok) throw new Error(`GitHub ${opts.method || 'GET'} ${path}: ${r.status} ${await r.text()}`);
  return opts.accept ? r.text() : r.status === 204 ? null : r.json();
};

async function callLLM(system, user) {
  const provider = (env.REVIEW_LLM_PROVIDER || 'openai').toLowerCase();
  const key = env.REVIEW_LLM_API_KEY; const model = env.REVIEW_LLM_MODEL;
  if (!key) throw new Error('REVIEW_LLM_API_KEY not set');
  if (!model) throw new Error('REVIEW_LLM_MODEL not set');
  const ctl = AbortSignal.timeout(120000);
  if (provider === 'anthropic') {
    const r = await fetch((env.REVIEW_LLM_BASE_URL || 'https://api.anthropic.com') + '/v1/messages', { method: 'POST', signal: ctl,
      headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
      body: JSON.stringify({ model, max_tokens: 800, temperature: 0, system, messages: [{ role: 'user', content: user }] }) });
    if (!r.ok) throw new Error(`anthropic ${r.status}`);
    const j = await r.json(); return (j.content || []).map(c => c.text || '').join('');
  }
  if (provider === 'openai') {
    const r = await fetch((env.REVIEW_LLM_BASE_URL || 'https://api.openai.com/v1') + '/chat/completions', { method: 'POST', signal: ctl,
      headers: { Authorization: `Bearer ${key}`, 'content-type': 'application/json' },
      body: JSON.stringify({ model, temperature: 0, response_format: { type: 'json_object' }, messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }) });
    if (!r.ok) throw new Error(`openai-compatible ${r.status}`);
    const j = await r.json(); return j.choices?.[0]?.message?.content;
  }
  throw new Error(`unknown REVIEW_LLM_PROVIDER ${provider}`);
}

(async () => {
  const cfg = L.parseConfig(fs.readFileSync(env.REVIEW_CONFIG || '.github/review-bot.yml', 'utf8'));
  const info = await gh(`/pulls/${pr}`);
  if (info.head.sha !== env.HEAD_SHA) { console.log('PR head moved; a newer run will review it.'); return; }
  const files = []; for (let p = 1; p <= 30; p++) { const b = await gh(`/pulls/${pr}/files?per_page=100&page=${p}`); files.push(...b); if (b.length < 100) break; }
  // Check outcomes come from the untrusted-code job via job outputs; anything but 'success' fails closed.
  const checks = {};
  for (const k of ['lint_python', 'lint_cpp', 'lint_mojo', 'validate']) checks[k] = env[`CHECK_${k.toUpperCase()}`];
  // Existing challenge folders, read from the BASE checkout (trusted).
  const existing = [];
  for (const d of ['easy', 'medium', 'hard']) {
    const root = `challenges/${d}`;
    if (fs.existsSync(root)) for (const n of fs.readdirSync(root)) if (fs.statSync(`${root}/${n}`).isDirectory()) existing.push(`${root}/${n}`);
  }
  const rules = L.ruleCheck(files, cfg, checks, new Set(existing));
  let llm = null;
  if (rules.safe) {
    try {
      const raw = await gh(`/pulls/${pr}`, { accept: 'application/vnd.github.v3.diff' });
      const t = L.truncateDiff(raw, cfg.max_diff_chars_for_llm || 60000);
      const out = await callLLM(L.SYSTEM_PROMPT, L.buildUserPrompt({ title: info.title, body: info.body, existing: existing.map(e => e.split('/').slice(1).join('/')), ...t }, crypto.randomBytes(12).toString('hex')));
      llm = L.parseVerdict(out) || { error: 'invalid JSON verdict from model' };
      if (t.truncated && !llm.error) llm = { ...llm, safe: false, reasons: ['diff truncated', ...llm.reasons] };
    } catch (e) { llm = { error: e.message }; }
  }
  const d = L.decide(rules, llm);
  const list = d.reasons.map(r => `- ${r.replace(/[@`]/g, m => m === '@' ? '@\u200b' : "'")}`).join('\n');
  const marker = `<!-- review-bot sha:${env.HEAD_SHA} -->`;
  if (d.approve) {
    await gh(`/pulls/${pr}/reviews`, { method: 'POST', body: JSON.stringify({ commit_id: env.HEAD_SHA, event: 'APPROVE', body: `${marker}\n✅ Review bot: low risk, safe to merge. Lint and structure checks passed; no sensitive paths or grader edits; ${rules.changedLines} lines changed.\n${list}` }) });
    console.log('Approved');
  } else {
    await gh(`/pulls/${pr}/reviews`, { method: 'POST', body: JSON.stringify({ commit_id: env.HEAD_SHA, event: 'COMMENT', body: `${marker}\n⚠️ Review bot: needs a human reviewer.\n${list}` }) });
    const reviewers = (cfg.human_reviewers || []).filter(r => r.toLowerCase() !== info.user.login.toLowerCase());
    if (reviewers.length) await gh(`/pulls/${pr}/requested_reviewers`, { method: 'POST', body: JSON.stringify({ reviewers }) }).catch(e => console.log(`::warning::${e.message}`));
    const label = cfg.human_label || 'needs-human-review';
    await gh(`/labels/${encodeURIComponent(label)}`).catch(() => gh('/labels', { method: 'POST', body: JSON.stringify({ name: label, color: 'd93f0b', description: 'Review bot: needs a human reviewer' }) })).catch(e => console.log(`::warning::${e.message}`));
    await gh(`/issues/${pr}/labels`, { method: 'POST', body: JSON.stringify({ labels: [label] }) }).catch(e => console.log(`::warning::${e.message}`));
    console.log('Needs human review:\n' + list);
  }
})().catch(e => { console.error(e); process.exit(1); });
