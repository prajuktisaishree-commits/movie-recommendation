/* app.js — CineMatch Frontend */

const API = "";  // Same-origin; change to http://localhost:5000 if running separately

// ──────────────────────────────────────────────
// Tabs
// ──────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function showLoader(container) {
  container.innerHTML = `<div class="loader"><div class="spinner"></div></div>`;
}

function showError(container, msg) {
  container.innerHTML = `<div class="state-msg error">⚠️ ${msg}</div>`;
}

function renderCards(container, movies, scoreKey, scoreLabel) {
  if (!movies || movies.length === 0) {
    container.innerHTML = `<div class="state-msg">No results found.</div>`;
    return;
  }
  container.innerHTML = movies.map((m, i) => {
    const score = m[scoreKey];
    const scoreHTML = score != null
      ? `<div class="card-score">${scoreLabel}: ${score}</div>`
      : "";
    const genres = (m.genres || []).map(g => `<span class="genre-tag">${g}</span>`).join("");
    return `
      <div class="movie-card" style="animation-delay:${i * 50}ms">
        <div class="card-rank">#${i + 1}</div>
        <div class="card-title">${m.title}</div>
        <div class="card-year">${m.year || ""}</div>
        <div class="card-genres">${genres}</div>
        ${scoreHTML}
      </div>`;
  }).join("");
}

// ──────────────────────────────────────────────
// Collaborative Filtering
// ──────────────────────────────────────────────
document.getElementById("cf-btn").addEventListener("click", async () => {
  const userId = document.getElementById("cf-user-id").value;
  const n = document.getElementById("cf-n").value;
  const container = document.getElementById("cf-results");
  showLoader(container);
  try {
    const res = await fetch(`${API}/api/recommend/collaborative?user_id=${userId}&n=${n}`);
    const data = await res.json();
    if (data.error) { showError(container, data.error); return; }
    renderCards(container, data.recommendations, "predicted_rating", "⭐ Predicted");
  } catch (e) {
    showError(container, "Could not reach the API. Is the Flask server running?");
  }
});

// ──────────────────────────────────────────────
// Content-Based — populate select
// ──────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch(`${API}/api/movies`);
    const data = await res.json();
    const sel = document.getElementById("cb-title");
    (data.movies || []).forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.title;
      opt.textContent = `${m.title} (${m.year})`;
      sel.appendChild(opt);
    });
  } catch (_) {}
})();

document.getElementById("cb-btn").addEventListener("click", async () => {
  const title = document.getElementById("cb-title").value;
  const n = document.getElementById("cb-n").value;
  const container = document.getElementById("cb-results");
  if (!title) { showError(container, "Please select a movie."); return; }
  showLoader(container);
  try {
    const res = await fetch(`${API}/api/recommend/content?title=${encodeURIComponent(title)}&n=${n}`);
    const data = await res.json();
    if (data.error) { showError(container, data.error); return; }
    renderCards(container, data.recommendations, "similarity_score", "🔗 Similarity");
  } catch (e) {
    showError(container, "Could not reach the API. Is the Flask server running?");
  }
});

// ──────────────────────────────────────────────
// Genre Chips
// ──────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch(`${API}/api/genres`);
    const data = await res.json();
    const wrap = document.getElementById("genre-chips");
    (data.genres || []).forEach(g => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = g;
      chip.addEventListener("click", () => chip.classList.toggle("selected"));
      wrap.appendChild(chip);
    });
  } catch (_) {}
})();

document.getElementById("g-btn").addEventListener("click", async () => {
  const selected = [...document.querySelectorAll(".chip.selected")].map(c => c.textContent);
  const n = document.getElementById("g-n").value;
  const container = document.getElementById("g-results");
  if (selected.length === 0) { showError(container, "Please select at least one genre."); return; }
  showLoader(container);
  try {
    const res = await fetch(`${API}/api/recommend/genre?genres=${encodeURIComponent(selected.join(","))}&n=${n}`);
    const data = await res.json();
    if (data.error) { showError(container, data.error); return; }
    renderCards(container, data.recommendations, null, null);
  } catch (e) {
    showError(container, "Could not reach the API. Is the Flask server running?");
  }
});
