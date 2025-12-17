// layout.jsx
import { NavLink, Outlet } from "react-router-dom";
import "./layout.css";

export default function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>🤖 RLT</h2>

        <nav>
          <div className="nav-section">Pipeline</div>
          <NavLink to="/train">📊 Import + Train</NavLink>

          <div className="nav-section">EDA</div>
          <NavLink to="/eda/understanding">🎯 Target & Missing Values</NavLink>
          <NavLink to="/eda/correlation">🔗 Correlation</NavLink>
          <NavLink to="/eda/boxplots">📦 Boxplots</NavLink>
          <NavLink to="/eda/top">⭐ Score Features</NavLink>

          <div className="nav-section">XAI</div>
          <NavLink to="/xai">🧠 Explainability</NavLink>
        </nav>
      </aside>

      <section className="content">
        <Outlet />
      </section>
    </div>
  );
}
