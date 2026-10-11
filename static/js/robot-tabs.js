// Robot tabs for the held-out videos. Every panel stays visible without JavaScript.
document.querySelectorAll('[data-robot-tabs]').forEach(tabList => {
  const tabs = Array.from(tabList.querySelectorAll('[role="tab"]'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));

  function selectTab(index, moveFocus = false) {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
      if (i !== index) panels[i].querySelectorAll('video').forEach(video => video.pause());
      // Posters of tabs not yet opened wait in data-poster, so hidden tabs cost nothing on page load.
      else panels[i].querySelectorAll('video[data-poster]').forEach(video => {
        video.poster = video.dataset.poster;
        video.removeAttribute('data-poster');
      });
    });
    if (moveFocus) tabs[index].focus();
  }

  tabs.forEach((tab, index) => {
    tab.id = `${tab.getAttribute('aria-controls')}-tab`;
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', tab.id);
    tab.addEventListener('click', () => selectTab(index));
    tab.addEventListener('keydown', event => {
      const destinations = {
        ArrowRight: (index + 1) % tabs.length,
        ArrowLeft: (index - 1 + tabs.length) % tabs.length,
        Home: 0,
        End: tabs.length - 1,
      };
      if (!Object.prototype.hasOwnProperty.call(destinations, event.key)) return;
      event.preventDefault();
      selectTab(destinations[event.key], true);
    });
  });
  selectTab(0);
  tabList.hidden = false;
});
