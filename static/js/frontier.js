// Frontier explorer: where each method's distinct tasks land after training, per robot.
// Task data (one file per robot) is loaded on demand; summary metrics come from summary.js.
(() => {
  const section = document.querySelector('#results');
  const explorer = section.querySelector('.frontier-explorer');
  const summary = window.FOSA_SUMMARY;
  if (!explorer || !summary) return;
  const tabs = Array.from(section.querySelectorAll('[data-frontier-tabs] [role="tab"]'));
  const chart = explorer.querySelector('[data-frontier-chart]');
  const status = explorer.querySelector('[data-frontier-status]');
  const detail = explorer.querySelector('[data-frontier-detail]');
  const metricsTitle = explorer.querySelector('[data-metrics-title]');
  const metricsTable = explorer.querySelector('[data-metrics-table]');
  const dataUrl = new URL('../data/', document.currentScript.src).href;
  const methods = Object.keys(summary.methods);
  const ours = 'fosa_learn';
  const cap = .3;  // bar height saturates at 30% of a method's tasks
  const shown = 6;
  let robot = 'jackal', selection = null, focusKey = null;

  const label = method => method === ours ? 'FOSA (ours)' : summary.methods[method];
  const fmt = (value, digits = 2) => value.toFixed(digits);
  const pct = value => `${Math.round(value * 100)}%`;
  const escape = text => text.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // Bin 0: never succeeds (J = 0). Bins 1-10: 0 < J < 1 in steps of 0.1. Bin 11: mastered (J = 1).
  const binOf = j => j <= 0 ? 0 : j >= 1 ? 11 : 1 + Math.min(9, Math.floor(j * 10));
  const binRange = bin => bin === 0 ? 'J = 0' : bin === 11 ? 'J = 1' : `${fmt((bin - 1) / 10, 1)} ≤ J < ${fmt(bin / 10, 1)}`;
  const binScore = bin => { if (bin === 0 || bin === 11) return 0; const c = (bin - .5) / 10; return 4 * c * (1 - c); };
  const bandColor = value => {
    const mix = (a, b) => Math.round(a + (b - a) * value);
    return `rgb(${mix(255, 184)},${mix(255, 165)},${mix(255, 212)})`;
  };

  function load(name) {
    if (window.FOSA_FRONTIER && window.FOSA_FRONTIER[name]) return Promise.resolve(window.FOSA_FRONTIER[name]);
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `${dataUrl}frontier-${name}.js`;
      script.onload = () => resolve(window.FOSA_FRONTIER[name]);
      script.onerror = reject;
      document.head.append(script);
    });
  }

  function binsFor(tasks) {
    const bins = Array.from({ length: 12 }, () => ({ tasks: [], frontier: 0 }));
    tasks.forEach(task => {
      const bin = bins[binOf(task[1])];
      bin.tasks.push(task);
      if (task[2] > 0) bin.frontier += 1;
    });
    return bins;
  }

  function cell(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function binGrid(className, build) {
    const grid = cell('div', className);
    for (let bin = 0; bin < 12; bin++) {
      grid.append(build(bin));
      if (bin === 0 || bin === 10) grid.append(cell('span', 'frontier-gap'));
    }
    return grid;
  }

  function renderChart(data) {
    chart.textContent = '';
    const band = binGrid('frontier-band', bin => {
      const swatch = cell('span');
      swatch.style.background = bandColor(binScore(bin));
      swatch.title = `Frontier score at ${binRange(bin)}, learned from J⁰ = 0: ${fmt(binScore(bin))}`;
      return swatch;
    });
    band.setAttribute('aria-hidden', 'true');
    chart.append(cell('span', 'frontier-band-label', 'frontier score'), band);

    methods.forEach((method, row) => {
      const tasks = data.methods[method];
      const bins = binsFor(tasks);
      const [prox, best] = summary.radar[robot][method][0];
      const group = cell('div', `frontier-row${method === ours ? ' is-ours' : ''}`);
      group.setAttribute('role', 'group');
      group.setAttribute('aria-label', `${label(method)}: ${tasks.length} distinct tasks, Frontier-Prox ${fmt(prox, 3)}`);
      const name = cell('div', 'frontier-row-label');
      name.innerHTML = `<b>${label(method)}</b><span>Frontier-Prox ${best ? `<strong>${fmt(prox, 3)}</strong>` : fmt(prox, 3)}</span>`;
      const grid = binGrid('frontier-bins', bin => {
        const { tasks: inBin, frontier } = bins[bin];
        const share = inBin.length / tasks.length;
        const button = cell('button', 'frontier-bin');
        button.type = 'button';
        button.dataset.key = `${method}:${bin}`;
        button.dataset.row = String(row);
        button.dataset.bin = String(bin);
        button.tabIndex = -1;
        button.setAttribute('aria-pressed', 'false');
        button.setAttribute('aria-label', `${label(method)}, ${binRange(bin)}: ${inBin.length} of ${tasks.length} tasks (${pct(share)}), ${frontier} on the frontier`);
        if (!inBin.length) { button.disabled = true; return button; }
        const scale = Math.min(1, cap / share) / cap;
        const other = cell('span', 'seg-other');
        other.style.height = `${(inBin.length - frontier) / tasks.length * scale * 100}%`;
        const front = cell('span', 'seg-frontier');
        front.style.height = `${frontier / tasks.length * scale * 100}%`;
        button.append(other, front);
        if (share > cap) button.append(cell('i', 'cap-mark'), cell('b', 'overflow', pct(share)));
        button.addEventListener('click', () => select(method, bin));
        return button;
      });
      group.append(name, grid);
      chart.append(group);
    });

    const axis = binGrid('frontier-axis', bin => {
      if (bin === 0) return cell('span', '', '0');
      if (bin === 11) return cell('span', '', '1');
      if (bin === 1) return cell('span');
      return cell('span', `tick-edge${bin % 2 ? '' : ' tick-minor'}`, fmt((bin - 1) / 10, 1).replace(/^0/, ''));
    });
    axis.setAttribute('aria-hidden', 'true');
    chart.append(cell('span'), axis, cell('span'),
      cell('p', 'frontier-axis-title', `Success after training, J · bar height = share of each method's distinct tasks (capped at ${pct(cap)})`));
  }

  function buttons() { return Array.from(chart.querySelectorAll('.frontier-bin:not(:disabled)')); }

  function onKey(event) {
    const current = event.target.closest('.frontier-bin');
    if (!current) return;
    const all = buttons();
    const row = Number(current.dataset.row), bin = Number(current.dataset.bin);
    const inRow = all.filter(b => Number(b.dataset.row) === row);
    let next;
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      const i = inRow.indexOf(current) + (event.key === 'ArrowRight' ? 1 : -1);
      next = inRow[(i + inRow.length) % inRow.length];
    } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      const step = event.key === 'ArrowDown' ? 1 : -1;
      for (let r = row + step; r >= 0 && r < methods.length && !next; r += step) {
        const candidates = all.filter(b => Number(b.dataset.row) === r);
        next = candidates.reduce((a, b) => Math.abs(b.dataset.bin - bin) < Math.abs(a.dataset.bin - bin) ? b : a, candidates[0]);
      }
    } else if (event.key === 'Home') next = inRow[0];
    else if (event.key === 'End') next = inRow[inRow.length - 1];
    if (!next) return;
    event.preventDefault();
    select(methods[next.dataset.row], Number(next.dataset.bin), true);
  }

  function skeletonHtml(data, skeleton) {
    if (!skeleton.length) return '<p class="task-freeform">Free-form task proposed by the LLM curriculum; no symbolic skeleton.</p>';
    return `<div class="task-skeleton">${skeleton.map(([atom, args]) => {
      const [role, name] = data.atoms[atom];
      return `<span><b>${role}</b> ${escape(name)}${args ? ` (${escape(args)})` : ''}</span>`;
    }).join('')}</div>`;
  }

  function renderDetail(data, method, bin, limit) {
    const inBin = binsFor(data.methods[method])[bin].tasks.slice().sort((a, b) => b[2] - a[2] || b[1] - a[1]);
    const frontier = inBin.filter(task => task[2] > 0).length;
    detail.innerHTML = `
      <h3>${label(method)} · ${summary.robots[robot]} · ${binRange(bin)}</h3>
      <p class="frontier-detail-summary">${inBin.length} distinct task${inBin.length === 1 ? '' : 's'}, ${frontier} on the frontier. Sorted by frontier score 𝓕.</p>
      <ul class="task-list">${inBin.slice(0, limit).map(([j0, j, f, runs, description, skeleton]) => `
        <li>
          <div class="task-scores"><span>J⁰ ${fmt(j0)} → J ${fmt(j)}</span><span>${f > 0 ? `<strong>𝓕 ${fmt(f)}</strong>` : '𝓕 0'}</span>${runs > 1 ? `<span>${runs} runs</span>` : ''}</div>
          ${escape(description)}
          ${skeletonHtml(data, skeleton)}
        </li>`).join('')}
      </ul>`;
    if (inBin.length > limit) {
      const more = cell('button', 'task-more', `Show all ${inBin.length} tasks`);
      more.type = 'button';
      more.addEventListener('click', () => renderDetail(data, method, bin, Infinity));
      detail.append(more);
    }
  }

  function select(method, bin, moveFocus = false) {
    selection = { method, bin };
    focusKey = `${method}:${bin}`;
    buttons().forEach(button => {
      const on = button.dataset.key === focusKey;
      button.setAttribute('aria-pressed', String(on));
      button.tabIndex = on ? 0 : -1;
      if (on && moveFocus) button.focus();
    });
    renderDetail(window.FOSA_FRONTIER[robot], method, bin, shown);
  }

  function defaultSelection(data) {
    const mass = binsFor(data.methods[ours]).map(({ tasks }) => tasks.reduce((sum, task) => sum + task[2], 0));
    return { method: ours, bin: mass.indexOf(Math.max(...mass)) };
  }

  function renderMetrics() {
    metricsTitle.textContent = `All discovery metrics · ${summary.robots[robot]}`;
    const robotIndex = Object.keys(summary.robots).indexOf(robot);
    const head = [...summary.axes, 'Frontier allocation'].map(axis => `<th scope="col">${axis}</th>`).join('');
    const rows = methods.map((method, i) => {
      const values = summary.radar[robot][method].map(([value, best]) => `<td${best ? ' class="is-best"' : ''}>${fmt(value, 3)}</td>`).join('');
      const alloc = summary.alloc[method];
      const allocCell = alloc ? `<td${alloc[robotIndex][1] ? ' class="is-best"' : ''}>${fmt(alloc[robotIndex][0])}×</td>` : '<td>—</td>';
      const group = i === 0 ? '<tr class="group-row"><th scope="rowgroup" colspan="8">Baselines</th></tr>'
        : i === 4 ? '<tr class="group-row"><th scope="rowgroup" colspan="8">Ablations</th></tr>' : '';
      return `${group}<tr${method === ours ? ' class="is-ours"' : ''}><th scope="row">${label(method)}</th>${values}${allocCell}</tr>`;
    }).join('');
    metricsTable.innerHTML = `
      <table class="results-table" aria-describedby="metrics-note">
        <thead><tr><th scope="col">Method</th>${head}</tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function selectRobot(name, moveFocus = false) {
    robot = name;
    tabs.forEach(tab => {
      const on = tab.dataset.robot === name;
      tab.setAttribute('aria-selected', String(on));
      tab.tabIndex = on ? 0 : -1;
      if (on) { explorer.setAttribute('aria-labelledby', tab.id); if (moveFocus) tab.focus(); }
    });
    renderMetrics();
    status.hidden = false;
    status.textContent = 'Loading discovered tasks…';
    load(name).then(data => {
      if (robot !== name) return;
      status.hidden = true;
      renderChart(data);
      const { method, bin } = defaultSelection(data);
      select(method, bin);
    }).catch(() => { status.textContent = 'The discovered tasks could not be loaded.'; });
  }

  tabs.forEach((tab, index) => {
    tab.id = `frontier-${tab.dataset.robot}-tab`;
    tab.addEventListener('click', () => selectRobot(tab.dataset.robot));
    tab.addEventListener('keydown', event => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      selectRobot(tabs[(index + step + tabs.length) % tabs.length].dataset.robot, true);
    });
  });
  chart.addEventListener('keydown', onKey);
  section.querySelector('[data-frontier-tabs]').hidden = false;
  explorer.hidden = false;
  selectRobot(robot);
})();
