/* ============================================================================
   ShopCare 前端主逻辑 (frontend/web/js/app.js)

   零构建、零依赖: 原生 ES2020 + fetch + canvas 手绘图表.
   为什么不用 Chart.js: 设计文档里写了用 CDN, 但这个项目要求能在**完全离线**的
   机器上跑. 手绘三个基础图表(横向柱 / 环形 / 折线)大约 120 行, 换来的是"断网也能演示".

   状态管理很简单: 一个 state 对象 + 若干 render 函数.
   规模再大就该上框架了, 但这个页面的状态量(令牌 / 模型清单 / 标签元数据 / 当前结果)
   用原生写反而更好读.
   ============================================================================ */

'use strict';

const API = '/api/v1';

/* ------------------------------ 全局状态 ------------------------------ */
const state = {
  token: localStorage.getItem('shopcare_token') || '',
  user: JSON.parse(localStorage.getItem('shopcare_user') || 'null'),
  labelsMeta: [],        // 9 类标签元数据(来自 /labels)
  models: [],            // 模型清单(来自 /models)
  defaultModel: '',
  result: null,          // 最近一次分类结果
  compareRows: [],
  reviewPage: 1,
  reviewTicket: null,
  charts: {},            // 图表缓存, 便于窗口缩放时重绘
  lastOverview: null,
};

/* 示例工单: 前三条是同分布语料, 后面几条覆盖"语料外新说法"与边界情况.
   最后一条用来演示拒识(模型应当不给任何标签). */
const SAMPLES = [
  { text: '快递到广州十天了还没动静, 客服也不回复, 我要退款', note: '物流+售后+服务(负面)' },
  { text: '快递一直没到, 客服也不回复', note: '对照组差异用例' },
  { text: '收到货就是坏的, 屏幕有裂纹, 申请换货', note: '质量问题' },
  { text: '发票开错了, 抬头写成公司旧名字, 能重开吗', note: '发票' },
  { text: '支付时扣了两次钱, 订单只显示一笔, 赶紧退回', note: '支付(高优先级)' },
  // 下面两条是给"拒识"演示准备的, 注意别把语义搞反(实测结论):
  //   '随便逛逛' 的措辞与 invalid 模板高度重合 -> 模型会**明确判为 无效与恶意**, 不拒识;
  //   '在的吗'   9 个标签全都不过阈值        -> 拒识(双阈值链路的正例:
  //              rf 走 low_confidence, fasttext 走 no_label_activated).
  { text: '随便逛逛', note: '命中 invalid 标签(不是拒识)' },
  { text: '在的吗', note: '无有效诉求(演示拒识: 双阈值都没过)' },
];

/* ------------------------------ 小工具 ------------------------------ */
const $ = (id) => document.getElementById(id);

function show(id, visible) {
  const node = $(id);
  if (node) node.hidden = !visible;
}

function setText(id, text) {
  const node = $(id);
  if (node) node.textContent = text;
}

function setHtml(id, html) {
  const node = $(id);
  if (node) node.innerHTML = html;
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function pct(value) {
  return (Number(value || 0) * 100).toFixed(1) + '%';
}

let toastTimer = null;
function toast(message, isError) {
  const node = $('toast');
  node.textContent = message;
  node.classList.toggle('err', !!isError);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 2600);
}

function showError(id, message) {
  const node = $(id);
  if (!node) return;
  node.textContent = message || '';
  node.hidden = !message;
}

/* ------------------------------ 接口封装 ------------------------------ */
async function api(path, options) {
  const opts = Object.assign({ method: 'GET', headers: {} }, options || {});
  opts.headers['Content-Type'] = 'application/json';
  if (state.token) opts.headers['Authorization'] = 'Bearer ' + state.token;
  if (opts.body && typeof opts.body !== 'string') opts.body = JSON.stringify(opts.body);

  let response;
  try {
    response = await fetch(API + path, opts);
  } catch (err) {
    throw new Error('无法连接后端(' + err.message + '), 请确认服务已启动');
  }

  let body = null;
  try { body = await response.json(); } catch (err) { body = null; }

  if (response.status === 401) {
    clearSession();
    showView('login');
    throw new Error((body && body.message) || '登录已过期, 请重新登录');
  }
  if (!response.ok || !body || body.code !== 0) {
    throw new Error((body && body.message) || ('请求失败: HTTP ' + response.status));
  }
  return body.data;
}

/* ------------------------------ 登录态 ------------------------------ */
function saveSession(data) {
  state.token = data.token;
  state.user = data.user;
  localStorage.setItem('shopcare_token', data.token);
  localStorage.setItem('shopcare_user', JSON.stringify(data.user));
  renderUserBox();
}

function clearSession() {
  state.token = '';
  state.user = null;
  localStorage.removeItem('shopcare_token');
  localStorage.removeItem('shopcare_user');
  renderUserBox();
}

function renderUserBox() {
  const logged = !!state.token;
  setText('top-user', logged ? (state.user.username + '(' + state.user.role + ')') : '未登录');
  show('btn-logout', logged);
  document.querySelectorAll('.nav-item[data-view]:not([data-view=login]):not([data-view=system])')
    .forEach((btn) => { btn.disabled = !logged; });
  const loginNav = document.querySelector('.nav-item[data-view=login]');
  if (loginNav) loginNav.textContent = logged ? '账号' : '登录 / 注册';
}

/* ------------------------------ 视图切换 ------------------------------ */
function showView(name) {
  if (name !== 'login' && name !== 'system' && !state.token) {
    toast('请先登录', true);
    name = 'login';
  }
  document.querySelectorAll('.view').forEach((node) => node.classList.remove('active'));
  const view = $('view-' + name);
  if (view) view.classList.add('active');
  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === name);
  });
  if (name === 'review') loadReviewQueue();
  if (name === 'dashboard') loadDashboard();
  if (name === 'system') loadSystem();
}

/* ------------------------------ 登录视图 ------------------------------ */
async function doLogin(event) {
  event.preventDefault();
  showError('login-error', '');
  const username = $('login-username').value.trim();
  const password = $('login-password').value;
  try {
    const data = await api('/user/login', { method: 'POST', body: { username, password } });
    saveSession(data);
    toast('欢迎回来, ' + data.user.username);
    await bootAfterLogin();
    showView('workbench');
  } catch (err) {
    showError('login-error', err.message);
  }
}

async function doRegister() {
  showError('login-error', '');
  const username = $('login-username').value.trim();
  const password = $('login-password').value;
  if (username.length < 3 || password.length < 6) {
    showError('login-error', '用户名至少 3 位, 密码至少 6 位');
    return;
  }
  try {
    await api('/user/register', { method: 'POST', body: { username, password } });
    toast('注册成功, 正在自动登录');
    await doLogin(new Event('submit'));
  } catch (err) {
    showError('login-error', err.message);
  }
}

async function doLogout() {
  try { await api('/user/logout', { method: 'POST' }); } catch (err) { /* 登出失败也要清本地 */ }
  clearSession();
  showView('login');
  toast('已登出');
}

/* ------------------------------ 工作台 ------------------------------ */
function renderSamples() {
  const box = $('wb-sample-list');
  box.innerHTML = SAMPLES.map((item, index) =>
    '<button class="btn btn-xs sample-btn" data-sample="' + index + '" title="' +
    escapeHtml(item.note) + '">' + escapeHtml(item.text.slice(0, 14)) + '...</button>'
  ).join('');
  box.querySelectorAll('button[data-sample]').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('wb-text').value = SAMPLES[Number(btn.dataset.sample)].text;
    });
  });
}

function renderModelOptions() {
  const select = $('wb-model');
  const options = [];
  state.models.forEach((model) => {
    const suffix = model.available ? '' : ' (未就绪)';
    options.push('<option value="' + model.key + '">' + escapeHtml(model.cn + suffix) + '</option>');
  });
  (state.models.virtual || []).forEach((extra) => {
    options.push('<option value="' + extra.key + '">' + escapeHtml(extra.cn) + '</option>');
  });
  select.innerHTML = options.join('');
  const preferred = state.defaultModel && state.models.some((m) => m.key === state.defaultModel && m.available)
    ? state.defaultModel
    : (state.models.find((m) => m.available) || { key: 'auto' }).key;
  select.value = preferred;
}

async function loadModelsAndLabels() {
  const models = await api('/models');
  state.defaultModel = models.default;
  state.models = models.items;
  state.models.virtual = models.virtual_choices || [];
  renderModelOptions();
  setText('top-model-chip', '模型: ' + (models.default || '-'));

  const labels = await api('/labels');
  state.labelsMeta = labels.labels;
}

function badge(text, kind) {
  return '<span class="badge badge-' + kind + '">' + escapeHtml(text) + '</span>';
}

function renderResult(result) {
  state.result = result;
  show('wb-placeholder', false);
  show('wb-result', true);

  /* --- 决策徽章 --- */
  const badges = [];
  if (result.rejected) {
    badges.push(badge(result.llm_fallback ? '拒识 · 已转 LLM' : '拒识 · 待人工复核', 'danger'));
  } else {
    badges.push(badge(result.resolved_by === 'llm' ? 'LLM 兜底分流' : '自动分流', 'ok'));
  }
  badges.push(badge('模型 ' + (result.model_cn || result.model_used), 'info'));
  if (result.fallback_from) badges.push(badge('降级自 ' + result.fallback_from, 'warn'));
  if (result.need_human_review) badges.push(badge('需人工复核', 'warn'));
  if ((result.suggested_reply || {}).needs_approval) badges.push(badge('回复需审批', 'warn'));
  if (result.cached) badges.push(badge('命中缓存', 'mute'));
  if (result.ticket_id) badges.push(badge('工单 ' + result.ticket_id, 'mute'));
  setHtml('wb-badges', badges.join(''));

  /* --- 标签胶囊 --- */
  const pills = (result.labels || []).map((item) =>
    '<span class="pill"><strong>' + escapeHtml(item.cn) + '</strong>' +
    '<span class="pct">' + pct(item.score) + '</span>' +
    '<span class="dept">' + escapeHtml(item.dept || '') + '</span></span>');
  setHtml('wb-labels', pills.length ? pills.join('')
    : '<span class="muted small">未激活任何标签(已按拒识处理)</span>');

  /* --- 9 类置信度条 ---
     数据源必须是 all_scores(全部 9 类的分数), 而不是 confidences:
     confidences 只包含**被激活**的标签, 未激活的一律缺失 —— 拿它画图的话,
     8 条柱子永远是空的, 看不出"模型在犹豫什么", 标题就成了假话。
     LLM 直连不产生 9 类分布(all_scores 为空), 这时退回到 confidences。 */
  const confidences = result.confidences || {};
  const allScores = (result.all_scores && Object.keys(result.all_scores).length)
    ? result.all_scores : confidences;
  const active = new Set((result.labels || []).map((item) => item.label));
  const order = state.labelsMeta.length
    ? state.labelsMeta
    : Object.keys(allScores).map((key) => ({ label: key, cn: key }));
  setHtml('wb-bars', order.map((meta) => {
    const score = Number(allScores[meta.label] || 0);
    const on = active.has(meta.label);
    return '<div class="bar-row' + (on ? ' on' : '') + '">' +
      '<span class="bar-name" title="' + escapeHtml(meta.cn) + '">' + escapeHtml(meta.cn) + '</span>' +
      '<span class="bar-track"><span class="bar-fill" style="width:' + (score * 100).toFixed(1) + '%"></span></span>' +
      '<span class="bar-val">' + pct(score) + '</span></div>';
  }).join(''));

  /* --- 关键字段 --- */
  const sentiment = result.sentiment || {};
  const priority = result.priority || {};
  const kv = [
    ['情感极性', (sentiment.cn || '-') + '(' + (sentiment.score || 0).toFixed(2) + ')'],
    ['优先级', (priority.cn || '-') + ' / ' + (priority.priority || '-')],
    ['要求响应', (result.sla_hours || '-') + ' 小时内首次响应'],
    ['建议部门', result.dept || '未分配'],
    ['平均置信度', pct(result.avg_confidence)],
    ['处理方', result.resolved_by || '-'],
    ['模型耗时', (result.model_latency_ms || 0).toFixed(1) + ' ms'],
    ['全链路耗时', (result.latency_ms || 0).toFixed(1) + ' ms'],
  ];
  setHtml('wb-kv', kv.map(([key, value]) =>
    '<div class="kv"><span class="kv-key">' + escapeHtml(key) + '</span>' +
    '<span class="kv-val">' + escapeHtml(value) + '</span></div>').join(''));

  /* --- 话术 --- */
  const reply = result.suggested_reply || {};
  setText('wb-reply', reply.reply || '(本模型/本次请求未生成话术)');

  /* --- 决策链路 --- */
  const trace = (result.trace || []).map((item) =>
    '<li><span class="step">' + escapeHtml(item.step) + '</span>: ' +
    '<span class="detail">' + escapeHtml(item.detail) + '</span></li>');
  if (result.kg && result.kg.summary) {
    trace.push('<li><span class="step">知识图谱</span>: <span class="detail">' +
      escapeHtml(result.kg.summary) + '</span></li>');
  }
  setHtml('wb-trace', trace.join('') || '<li class="muted">-</li>');
}

async function doClassify() {
  const text = $('wb-text').value.trim();
  showError('wb-error', '');
  if (!text) { showError('wb-error', '请输入工单文本'); return; }
  const save = $('wb-save').checked;
  $('wb-submit').disabled = true;
  setText('wb-submit', '推理中...');
  try {
    const result = await api('/classify', {
      method: 'POST',
      body: { text, model: $('wb-model').value, use_llm_fallback: $('wb-llm').checked,
              save, top_k: 3 },
    });
    renderResult(result);
    toast(save ? ('已建单 ' + result.ticket_id) : '分类完成');
  } catch (err) {
    showError('wb-error', err.message);
  } finally {
    $('wb-submit').disabled = false;
    setText('wb-submit', '提交分类');
  }
}

async function doCompare() {
  const text = $('wb-text').value.trim();
  showError('wb-error', '');
  if (!text) { showError('wb-error', '请先输入工单文本'); return; }
  const local = state.models.filter((model) => model.available).map((model) => model.key);
  if (!local.length) { showError('wb-error', '没有可用的本地模型'); return; }

  const button = $('wb-compare');
  button.disabled = true;
  setText('wb-compare', '对比中...');
  try {
    const rows = [];
    for (const key of local) {
      try {
        const result = await api('/classify', {
          method: 'POST',
          body: { text, model: key, use_llm_fallback: false, save: false, top_k: 3 },
        });
        rows.push({ key, result, error: null });
      } catch (err) {
        rows.push({ key, result: null, error: err.message });
      }
    }
    renderCompare(rows);
    toast('对比完成: ' + rows.length + ' 个模型');
  } finally {
    button.disabled = false;
    setText('wb-compare', '三模型对比');
  }
}

function renderCompare(rows) {
  state.compareRows = rows;
  setHtml('wb-compare-body', rows.map((row) => {
    const name = ($('wb-model').querySelector('option[value="' + row.key + '"]') || {}).textContent || row.key;
    if (!row.result) {
      return '<tr><td>' + escapeHtml(name) + '</td><td colspan="5" class="muted">' +
        escapeHtml(row.error || '不可用') + '</td></tr>';
    }
    const result = row.result;
    const labels = (result.labels || []).length
      ? result.labels.map((item) => escapeHtml(item.cn) + ' <span class="muted">' +
          pct(item.score) + '</span>').join('<br>')
      : '<span class="muted">无(拒识)</span>';
    const reject = result.rejected
      ? '<span class="badge badge-danger">是</span>'
      : '<span class="badge badge-ok">否</span>';
    const prio = result.priority || {};
    return '<tr><td><strong>' + escapeHtml(name) + '</strong></td><td>' + labels + '</td>' +
      '<td>' + reject + '</td><td>' + escapeHtml((result.sentiment || {}).cn || '-') + '</td>' +
      '<td>' + escapeHtml((prio.cn || '-') + ' / ' + (prio.priority || '-')) + '</td>' +
      '<td>' + (result.latency_ms || 0).toFixed(0) + ' ms</td></tr>';
  }).join(''));
  show('wb-compare-card', true);
  $('wb-compare-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ------------------------------ 人工复核 ------------------------------ */
async function loadReviewQueue() {
  showError('rv-error', '');
  const status = $('rv-status').value;
  try {
    const data = await api('/dashboard/review_queue?status=' + encodeURIComponent(status) +
                           '&page=' + state.reviewPage + '&size=10');
    setText('rv-summary', '共 ' + data.total + ' 条 · 本页需重点关注 ' + data.attention_count +
                          ' 条 · 第 ' + data.page + '/' + Math.max(data.pages, 1) + ' 页');
    setHtml('rv-body', (data.items || []).map((item) => {
      const labels = (item.labels || []).map((one) =>
        escapeHtml(one.cn || one)) .join('、') || '<span class="muted">无</span>';
      const flag = item.needs_attention
        ? ' <span class="badge badge-warn">关注</span>' : '';
      return '<tr><td class="mono">' + escapeHtml(item.ticket_id) + '</td>' +
        '<td class="text-cell"><span class="clamp" title="' + escapeHtml(item.text) + '">' +
          escapeHtml(item.text) + '</span></td>' +
        '<td>' + labels + flag + '</td>' +
        '<td>' + (item.priority ? '<span class="badge badge-p' + item.priority.slice(1) + '">' +
          escapeHtml(item.priority) + '</span>' : '-') + '</td>' +
        '<td class="mono">' + pct(item.avg_confidence) + '</td>' +
        '<td>' + escapeHtml(item.resolved_by || '-') + '</td>' +
        '<td class="muted small">' + escapeHtml(item.created_at || '-') + '</td>' +
        '<td><button class="btn btn-xs" data-review="' + escapeHtml(item.ticket_id) + '">复核</button></td></tr>';
    }).join('') || '<tr><td colspan="8" class="muted">队列为空</td></tr>');

    $('rv-body').querySelectorAll('button[data-review]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const item = (data.items || []).find((one) => one.ticket_id === btn.dataset.review);
        openReviewPanel(item);
      });
    });
    setText('rv-page-info', data.page + ' / ' + Math.max(data.pages, 1));
  } catch (err) {
    showError('rv-error', err.message);
  }
}

function openReviewPanel(ticket) {
  state.reviewTicket = ticket;
  show('rv-panel', true);
  setText('rv-panel-id', ticket.ticket_id);
  setText('rv-panel-text', ticket.text);
  const current = new Set((ticket.labels || []).map((one) => one.label || one));
  setHtml('rv-panel-labels', state.labelsMeta.map((meta) =>
    '<label><input type="checkbox" value="' + meta.label + '"' +
    (current.has(meta.label) ? ' checked' : '') + '> ' + escapeHtml(meta.cn) + '</label>').join(''));
  $('rv-panel-note').value = '';
  showError('rv-panel-error', '');
  $('rv-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function selectedReviewLabels() {
  return Array.from($('rv-panel-labels').querySelectorAll('input:checked'))
    .map((node) => node.value);
}

async function submitFeedback(action) {
  const ticket = state.reviewTicket;
  if (!ticket) return;
  showError('rv-panel-error', '');
  const labels = selectedReviewLabels();
  if (action === 'correct' && !labels.length) {
    showError('rv-panel-error', '请至少勾选一个正确标签; 若确实是无效内容, 请勾选「无效/骚扰」');
    return;
  }
  try {
    await api('/feedback', {
      method: 'POST',
      body: { ticket_id: ticket.ticket_id, action, corrected_labels: labels,
              note: $('rv-panel-note').value.trim() || null },
    });
    toast('复核已记录');
    show('rv-panel', false);
    state.reviewTicket = null;
    loadReviewQueue();
  } catch (err) {
    showError('rv-panel-error', err.message);
  }
}

/* ------------------------------ 数据看板 ------------------------------ */
function statCard(key, value, sub) {
  return '<div class="stat"><div class="stat-key">' + escapeHtml(key) + '</div>' +
    '<div class="stat-val">' + escapeHtml(value) + '</div>' +
    '<div class="stat-sub">' + escapeHtml(sub || '') + '</div></div>';
}

async function loadDashboard() {
  try {
    const info = await api('/dashboard/overview');
    state.lastOverview = info;
    // 中文名映射: 图表和表格里都直接显示中文, 不让人回头对照标签表
    const cnOf = {};
    state.labelsMeta.forEach((meta) => { cnOf[meta.label] = meta.cn; });
    setHtml('db-stats',
      statCard('工单总量', info.total, '存储: ' + info.storage_mode) +
      statCard('自动分流率', pct(info.auto_rate), '未拒识即自动分流') +
      statCard('拒识率', pct(info.reject_rate), '转 LLM / 人工') +
      statCard('平均耗时', (info.avg_latency_ms || 0).toFixed(1) + ' ms', '全链路(含规则)') +
      statCard('待复核', info.pending_review, '状态为 pending'));

    setHtml('db-top-labels', (info.top_labels || []).map((item) =>
      '<tr><td>' + escapeHtml(cnOf[item.label] || item.label) + ' <span class="muted small">' +
      escapeHtml(item.label) + '</span></td><td>' + item.count + '</td></tr>').join('')
      || '<tr><td colspan="2" class="muted">暂无数据</td></tr>');
    setHtml('db-reject-reasons', Object.entries(info.reject_reason_dist || {}).map(([key, value]) =>
      '<tr><td class="mono">' + escapeHtml(key) + '</td><td>' + value + '</td></tr>').join('')
      || '<tr><td colspan="2" class="muted">暂无数据</td></tr>');

    const labelItems = (info.top_labels || []).map((item) => ({
      key: cnOf[item.label] || item.label, value: item.count,
    }));
    drawBarChart($('db-chart-labels'), labelItems);

    const extra = [];
    Object.entries(info.priority_dist || {}).forEach(([key, value]) => extra.push(['优先级', key, value]));
    Object.entries(info.dept_dist || {}).forEach(([key, value]) => extra.push(['处理部门', key, value]));
    Object.entries(info.sentiment_dist || {}).forEach(([key, value]) => extra.push(['情感', key, value]));
    setHtml('db-dist-extra', extra.map((row) =>
      '<tr><td class="muted small">' + escapeHtml(row[0]) + '</td><td>' + escapeHtml(row[1]) +
      '</td><td>' + row[2] + '</td></tr>').join('')
      || '<tr><td colspan="3" class="muted">暂无数据</td></tr>');

    const modelItems = Object.entries(info.model_dist || {}).map(([key, value]) => ({ key, value }));
    drawDonutChart($('db-chart-models'), modelItems);

    const trend = info.trend || [];
    drawLineChart($('db-chart-trend'), trend.map((item) => item.date.slice(5)),
                  trend.map((item) => item.count));
  } catch (err) {
    toast('看板加载失败: ' + err.message, true);
  }
}

/* ------------------------------ 系统状态 ------------------------------ */
async function loadSystem() {
  try {
    const response = await fetch('/health');
    const body = await response.json();
    const info = body.data;
    setHtml('sys-stats',
      statCard('状态', info.status, info.healthy ? '存在可用模型' : '无可用模型') +
      statCard('存储', info.storage.storage_mode, info.storage.degraded ? '已降级, 数据不持久' : 'MySQL') +
      statCard('MySQL', info.storage.mysql.available ? '已连接' : '不可用',
               info.storage.mysql.error ? '降级原因见告警' : '') +
      statCard('Redis', info.storage.redis.available ? '已连接' : '不可用',
               info.storage.redis.error ? '已退化为进程内实现' : '') +
      statCard('可用模型', (info.models.available || []).join(', ') || '无', '默认: ' + info.models.default_model));

    setHtml('sys-models', (info.model_catalog || []).map((model) =>
      '<tr><td><strong>' + escapeHtml(model.cn) + '</strong><div class="muted small">' +
        escapeHtml(model.key) + '</div></td>' +
      '<td class="small">' + escapeHtml(model.desc || '') + '</td>' +
      '<td class="mono small">' + escapeHtml((model.model_path || '').split('\\').pop()) + '</td>' +
      '<td>' + (model.available ? '<span class="badge badge-ok">可用</span>'
                                : '<span class="badge badge-danger">不可用</span>') + '</td>' +
      '<td>' + (model.loaded ? '是' : '否') + '</td>' +
      '<td class="small muted">' + escapeHtml(model.load_error || model.extra_hint || '-') + '</td></tr>')
      .join(''));

    setHtml('sys-modules', Object.entries(info.business_modules || {}).map(([key, ok]) =>
      '<span class="pill">' + escapeHtml(key) + ' ' +
      (ok ? '<span class="badge badge-ok">就绪</span>'
          : '<span class="badge badge-warn">缺失</span>') + '</span>').join(''));

    const warnings = info.warnings || [];
    setHtml('sys-warnings', warnings.length
      ? warnings.map((text) => '<li>' + escapeHtml(text) + '</li>').join('')
      : '<li class="none">无告警</li>');
  } catch (err) {
    toast('系统状态获取失败: ' + err.message, true);
  }
}

/* ------------------------------ 手绘图表 ------------------------------ */
const PALETTE = ['#4356d6', '#1f9d6a', '#c98410', '#2b7fc4', '#d0453c',
                '#7b5cc4', '#0f9b9b', '#8a6d3b', '#9aa4b5'];

function fillRoundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
  ctx.fill();
}

function prepareCanvas(canvas, height) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement.clientWidth || 480;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function drawBarChart(canvas, items) {
  const { ctx, width, height } = prepareCanvas(canvas, 260);
  if (!items.length) { emptyChart(ctx, width, height); return; }
  const left = 74, right = 46, top = 8, bottom = 8;
  const plotWidth = width - left - right;
  const max = Math.max.apply(null, items.map((item) => item.value)) || 1;
  const rowHeight = Math.min(30, (height - top - bottom) / items.length);

  items.forEach((item, index) => {
    const y = top + index * rowHeight + rowHeight * 0.18;
    const barHeight = rowHeight * 0.62;
    const barWidth = Math.max(2, (item.value / max) * plotWidth);
    ctx.fillStyle = '#6b7789';
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(item.key.length > 6 ? item.key.slice(0, 6) : item.key, left - 8, y + barHeight / 2);
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    fillRoundRect(ctx, left, y, barWidth, barHeight, 4);
    ctx.fillStyle = '#3b465c';
    ctx.textAlign = 'left';
    ctx.fillText(String(item.value), left + barWidth + 6, y + barHeight / 2);
  });
}

function drawDonutChart(canvas, items) {
  const { ctx, width, height } = prepareCanvas(canvas, 260);
  if (!items.length) { emptyChart(ctx, width, height); return; }
  const total = items.reduce((sum, item) => sum + item.value, 0) || 1;
  const cx = Math.min(width * 0.32, 150), cy = height / 2;
  const outer = Math.min(height * 0.38, 88), inner = outer * 0.58;
  let angle = -Math.PI / 2;

  items.forEach((item, index) => {
    const slice = (item.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outer, angle, angle + slice);
    ctx.closePath();
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    ctx.fill();
    angle += slice;
  });
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.fillStyle = '#1c2434';
  ctx.font = '600 15px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(total), cx, cy - 8);
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = '#6b7789';
  ctx.fillText('次调用', cx, cy + 10);

  let legendY = height / 2 - items.length * 10;
  ctx.textAlign = 'left';
  items.forEach((item, index) => {
    ctx.fillStyle = PALETTE[index % PALETTE.length];
    fillRoundRect(ctx, width * 0.62, legendY - 5, 10, 10, 2);
    ctx.fillStyle = '#3b465c';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(item.key + ' · ' + item.value + ' (' + Math.round(item.value / total * 100) + '%)',
                 width * 0.62 + 16, legendY);
    legendY += 20;
  });
}

function drawLineChart(canvas, labels, values) {
  const { ctx, width, height } = prepareCanvas(canvas, 240);
  if (!values.length) { emptyChart(ctx, width, height); return; }
  const left = 34, right = 12, top = 12, bottom = 26;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const max = Math.max.apply(null, values.concat([1]));
  const stepX = values.length > 1 ? plotWidth / (values.length - 1) : 0;

  ctx.strokeStyle = '#e3e8f0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = top + (plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
    ctx.fillStyle = '#96a0b0';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(Math.round(max - (max / 4) * i)), left - 6, y);
  }

  const points = values.map((value, index) => ({
    x: left + stepX * index,
    y: top + plotHeight - (value / max) * plotHeight,
  }));

  ctx.beginPath();
  ctx.moveTo(points[0].x, top + plotHeight);
  points.forEach((point) => ctx.lineTo(point.x, point.y));
  ctx.lineTo(points[points.length - 1].x, top + plotHeight);
  ctx.closePath();
  ctx.fillStyle = 'rgba(67, 86, 214, .12)';
  ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => (index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y)));
  ctx.strokeStyle = '#4356d6';
  ctx.lineWidth = 2;
  ctx.stroke();

  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#4356d6';
    ctx.fill();
  });

  ctx.fillStyle = '#96a0b0';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  labels.forEach((label, index) => ctx.fillText(label, points[index].x, top + plotHeight + 6));
}

function emptyChart(ctx, width, height) {
  ctx.fillStyle = '#96a0b0';
  ctx.font = '13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('暂无数据', width / 2, height / 2);
}

/* ------------------------------ 复制 ------------------------------ */
async function copyReply() {
  const text = $('wb-reply').textContent || '';
  if (!text) return;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const area = document.createElement('textarea');
      area.value = text;
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      document.execCommand('copy');
      document.body.removeChild(area);
    }
    toast('话术已复制');
  } catch (err) {
    toast('复制失败, 请手动选择文本', true);
  }
}

/* ------------------------------ 启动 ------------------------------ */
async function bootAfterLogin() {
  try {
    await loadModelsAndLabels();
  } catch (err) {
    toast('模型/标签信息加载失败: ' + err.message, true);
  }
}

function bindEvents() {
  $('login-form').addEventListener('submit', doLogin);
  $('btn-register').addEventListener('click', doRegister);
  $('btn-logout').addEventListener('click', doLogout);

  document.querySelectorAll('.nav-item').forEach((btn) => {
    btn.addEventListener('click', () => showView(btn.dataset.view));
  });
  $('wb-submit').addEventListener('click', doClassify);
  $('wb-compare').addEventListener('click', doCompare);
  $('wb-clear').addEventListener('click', () => {
    $('wb-text').value = '';
    show('wb-result', false);
    show('wb-placeholder', true);
    show('wb-compare-card', false);
    showError('wb-error', '');
  });
  $('wb-compare-close').addEventListener('click', () => show('wb-compare-card', false));
  $('wb-copy').addEventListener('click', copyReply);

  $('rv-refresh').addEventListener('click', () => { state.reviewPage = 1; loadReviewQueue(); });
  $('rv-status').addEventListener('change', () => { state.reviewPage = 1; loadReviewQueue(); });
  $('rv-prev').addEventListener('click', () => {
    state.reviewPage = Math.max(1, state.reviewPage - 1);
    loadReviewQueue();
  });
  $('rv-next').addEventListener('click', () => {
    state.reviewPage += 1;
    loadReviewQueue();
  });
  $('rv-panel-correct').addEventListener('click', () => submitFeedback('correct'));
  $('rv-panel-approve').addEventListener('click', () => submitFeedback('approve'));
  $('rv-panel-close').addEventListener('click', () => {
    show('rv-panel', false);
    state.reviewTicket = null;
  });

  $('db-refresh').addEventListener('click', loadDashboard);
  $('sys-refresh').addEventListener('click', loadSystem);
  window.addEventListener('resize', () => {
    if (!$('view-dashboard').classList.contains('active')) return;
    loadDashboard();
  });
}

async function boot() {
  bindEvents();
  renderSamples();
  renderUserBox();
  loadSystem();

  if (state.token) {
    try {
      const profile = await api('/user/profile');
      state.user = { username: profile.username, role: profile.role };
      renderUserBox();
      await bootAfterLogin();
      showView('workbench');
      return;
    } catch (err) {
      clearSession();
    }
  }
  showView('login');
}

boot();