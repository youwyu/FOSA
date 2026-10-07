// Held-out transfer table, rendered from the same summary data as the explorer.
(() => {
  const target = document.querySelector('[data-heldout-table]');
  const summary = window.FOSA_SUMMARY;
  if (!target || !summary) return;
  const robots = Object.values(summary.robots);
  const rows = Object.keys(summary.methods).map((method, i) => {
    const { robots: cells, mean, above_half: aboveHalf } = summary.heldout[method];
    const ours = method === 'fosa_learn';
    const group = i === 0 ? '<tr class="group-row"><th scope="rowgroup" colspan="7">Baselines</th></tr>'
      : i === 4 ? '<tr class="group-row"><th scope="rowgroup" colspan="7">Ablations</th></tr>' : '';
    const values = cells.map(([value, low, high, best]) =>
      `<td${best ? ' class="is-best"' : ''}>${value.toFixed(3)}<span class="ci">[${low.toFixed(2)}, ${high.toFixed(2)}]</span></td>`).join('');
    return `${group}<tr${ours ? ' class="is-ours"' : ''}><th scope="row">${ours ? 'FOSA (ours)' : summary.methods[method]}</th>${values}<td>${mean.toFixed(3)}</td><td>${aboveHalf}</td></tr>`;
  }).join('');
  target.innerHTML = `
    <table class="results-table" aria-describedby="heldout-note">
      <thead><tr><th scope="col">Method</th>${robots.map(name => `<th scope="col">${name}</th>`).join('')}<th scope="col">Mean</th><th scope="col">Tasks &gt; 0.5</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
})();
