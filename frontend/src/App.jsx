
import {
  LayoutDashboard,
  Map,
  Route,
  ChartNoAxesCombined,
  Settings,
  Bell,
  Search,
  Activity,
  MapPin,
  Clock3,
  ArrowUpRight,
  Play,
  ChevronDown,
  Zap,
  Menu
} from "lucide-react";

import "./App.css";

function App() {
  return (
    <div className="app-container">

      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Route size={23} />
          </div>
          <div>
            <h2>PathSnap</h2>
            <span>MAP MATCHING PLATFORM</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          <a className="nav-item active">
            <LayoutDashboard size={19} />
            Dashboard
          </a>

          <a className="nav-item">
            <Map size={19} />
            Trip Explorer
          </a>

          <a className="nav-item">
            <Route size={19} />
            Route Reconstruction
          </a>

          <a className="nav-item">
            <ChartNoAxesCombined size={19} />
            Model Performance
          </a>
        </nav>

        <div className="sidebar-bottom">
          <a className="nav-item">
            <Settings size={19} />
            Settings
          </a>

          <div className="profile">
            <div className="avatar">T</div>
            <div>
              <strong>Talha</strong>
              <span>Frontend Developer</span>
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">

        {/* TOPBAR */}
        <header className="topbar">
          <div className="breadcrumb">
            <Menu className="mobile-menu" size={20} />
            <span>Workspace</span>
            <span>/</span>
            <strong>Dashboard</strong>
          </div>

          <div className="topbar-actions">
            <div className="search-box">
              <Search size={17} />
              <input placeholder="Search trips..." />
            </div>

            <button className="icon-button">
              <Bell size={19} />
            </button>

            <div className="avatar small">T</div>
          </div>
        </header>

        {/* DASHBOARD */}
        <section className="dashboard">

          <div className="welcome-section">
            <div>
              <div className="eyebrow">
                <Activity size={15} />
                GEOSPATIAL ANALYTICS
              </div>

              <h1>Route Intelligence Dashboard</h1>

              <p>
                Monitor GPS trajectories, reconstruct routes
                and evaluate map-matching performance.
              </p>
            </div>

            <div className="system-status">
              <span className="status-dot"></span>
              System ready
            </div>
          </div>

          {/* STATISTICS */}
          <div className="stats-grid">

            <div className="stat-card">
              <div className="stat-top">
                <span>Total Trips</span>
                <MapPin size={19} />
              </div>
              <h2>10</h2>
              <p>Available evaluation trips</p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Model Accuracy</span>
                <ChartNoAxesCombined size={19} />
              </div>
              <h2>—</h2>
              <p>Awaiting backend results</p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Routes Processed</span>
                <Route size={19} />
              </div>
              <h2>0</h2>
              <p>Predictions completed</p>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Inference Time</span>
                <Zap size={19} />
              </div>
              <h2>—</h2>
              <p>Awaiting model execution</p>
            </div>

          </div>

          {/* MAP AND TRIP PANEL */}
          <div className="content-grid">

            <div className="panel map-panel">
              <div className="panel-header">
                <div>
                  <h3>Geospatial Overview</h3>
                  <p>GPS trajectory and road network visualization</p>
                </div>

                <button className="secondary-button">
                  <Map size={16} />
                  Explore map
                </button>
              </div>

              <div className="map-placeholder">
                <div className="map-grid"></div>

                <div className="map-center">
                  <div className="map-icon">
                    <MapPin size={28} />
                  </div>
                  <h3>Interactive Map</h3>
                  <p>
                    Select a trip to visualize its GPS trajectory
                    and reconstructed route.
                  </p>
                </div>

                <div className="map-legend">
                  <span><i className="gps-dot"></i> GPS Points</span>
                  <span><i className="route-line"></i> Predicted Route</span>
                </div>
              </div>
            </div>

            {/* TRIP SELECTOR */}
            <div className="panel trip-panel">
              <div className="panel-header">
                <div>
                  <h3>Trip Explorer</h3>
                  <p>Select a trip to begin</p>
                </div>
                <MapPin size={20} />
              </div>

              <label className="field-label">SELECT TRIP</label>

              <div className="select-box">
                <span>Choose a trip</span>
                <ChevronDown size={18} />
              </div>

              <div className="trip-info">
                <div>
                  <span>GPS observations</span>
                  <strong>—</strong>
                </div>
                <div>
                  <span>Distance</span>
                  <strong>—</strong>
                </div>
                <div>
                  <span>Duration</span>
                  <strong>—</strong>
                </div>
              </div>

              <button className="primary-button" disabled>
                <Play size={17} />
                Run Reconstruction
              </button>

              <p className="helper-text">
                Backend connection pending
              </p>
            </div>

          </div>

          {/* PERFORMANCE */}
          <div className="panel performance-panel">
            <div className="panel-header">
              <div>
                <h3>Model Performance</h3>
                <p>Overview of route reconstruction metrics</p>
              </div>

              <span className="pending-badge">
                Awaiting evaluation
              </span>
            </div>

            <div className="performance-grid">
              <div>
                <span>Top-1 Accuracy</span>
                <h2>—</h2>
              </div>
              <div>
                <span>Macro F1 Score</span>
                <h2>—</h2>
              </div>
              <div>
                <span>Route Continuity</span>
                <h2>—</h2>
              </div>
              <div>
                <span>Inference Time</span>
                <h2>—</h2>
              </div>
            </div>
          </div>

          <footer>
            PathSnap © 2026 · Distributed Map-Matching System
            <span>ML Hackathon Project</span>
          </footer>

        </section>
      </main>
    </div>
  );
}

export default App;
