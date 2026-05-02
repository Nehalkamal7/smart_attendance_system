"""
Flask Web Dashboard for Smart Attendance System
Real-time attendance monitoring and analytics UI
Run with: python dashboard.py
Then open: http://127.0.0.1:5000
"""

import os
import json
import base64
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template_string, send_from_directory, request
from database_manager import AttendanceManager

app = Flask(__name__)
db = AttendanceManager()


# ── HTML TEMPLATE (inline for single-file deployment) ─────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Smart Attendance Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
/* ── RESET & BASE ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0b14;
  --bg2:#10121f;
  --card:#141726;
  --card2:#1a1f35;
  --border:#252a42;
  --accent:#63d4ff;
  --accent2:#8250ff;
  --accent3:#50e6b4;
  --success:#3cd678;
  --warning:#ffb832;
  --danger:#ff5050;
  --text:#e8ecf4;
  --muted:#6b7394;
  --font:'Inter',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;overflow-x:hidden}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}

/* ── LAYOUT ── */
.app{display:grid;grid-template-rows:auto 1fr;min-height:100vh}

/* ── HEADER ── */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 32px;height:64px;
  background:linear-gradient(90deg,#0d0f1e 0%,#111428 100%);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(16px);
}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{
  width:36px;height:36px;border-radius:10px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:18px;
}
.logo-text span{display:block;font-size:18px;font-weight:700;letter-spacing:-.3px}
.logo-text small{font-size:11px;color:var(--muted);font-weight:400}
.header-right{display:flex;align-items:center;gap:16px}
.live-badge{
  display:flex;align-items:center;gap:6px;padding:4px 12px;
  background:rgba(60,214,120,.12);border:1px solid rgba(60,214,120,.3);
  border-radius:999px;font-size:12px;color:var(--success);font-weight:500;
}
.live-dot{width:7px;height:7px;background:var(--success);border-radius:50%;animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.time-display{font-family:var(--mono);font-size:13px;color:var(--muted)}
.refresh-btn{
  padding:6px 14px;border-radius:8px;border:1px solid var(--border);
  background:var(--card);color:var(--text);font-size:12px;cursor:pointer;
  transition:all .2s;font-family:var(--font);
}
.refresh-btn:hover{border-color:var(--accent);color:var(--accent)}

/* ── MAIN CONTENT ── */
main{padding:28px 32px;display:flex;flex-direction:column;gap:28px}

/* ── KPI CARDS ── */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.kpi-card{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:20px 22px;position:relative;overflow:hidden;
  transition:border-color .25s,transform .25s;
}
.kpi-card:hover{border-color:var(--accent);transform:translateY(-2px)}
.kpi-card::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(99,212,255,.04) 0%,transparent 60%);
  pointer-events:none;
}
.kpi-icon{font-size:28px;margin-bottom:10px}
.kpi-label{font-size:12px;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.5px}
.kpi-value{font-size:36px;font-weight:800;font-family:var(--mono);margin:4px 0}
.kpi-sub{font-size:12px;color:var(--muted)}
.kpi-value.accent{color:var(--accent)}
.kpi-value.accent2{color:var(--accent2)}
.kpi-value.accent3{color:var(--accent3)}
.kpi-value.warn{color:var(--warning)}

/* ── BOTTOM GRID ── */
.bottom-grid{display:grid;grid-template-columns:1fr 360px;gap:20px;align-items:start}

/* ── TABLE CARD ── */
.table-card,.panel-card{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;overflow:hidden;
}
.card-header{
  padding:16px 22px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.card-title{font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}
.badge{
  padding:2px 10px;border-radius:999px;font-size:11px;font-weight:600;
  background:rgba(99,212,255,.12);color:var(--accent);
}
.badge.green{background:rgba(60,214,120,.12);color:var(--success)}
.badge.purple{background:rgba(130,80,255,.12);color:var(--accent2)}

table{width:100%;border-collapse:collapse}
thead th{
  padding:10px 16px;text-align:left;font-size:11px;font-weight:600;
  color:var(--muted);text-transform:uppercase;letter-spacing:.5px;
  border-bottom:1px solid var(--border);
}
tbody tr{transition:background .15s}
tbody tr:hover{background:rgba(255,255,255,.03)}
tbody td{
  padding:11px 16px;font-size:13px;border-bottom:1px solid rgba(37,42,66,.6);
}
tbody tr:last-child td{border-bottom:none}
.status-pill{
  display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:999px;font-size:11px;font-weight:600;
}
.status-pill.present{background:rgba(60,214,120,.12);color:var(--success)}
.method-tag{
  font-size:11px;padding:2px 8px;border-radius:6px;font-family:var(--mono);
  background:var(--card2);color:var(--muted);border:1px solid var(--border);
}
.confidence-bar{width:80px;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.confidence-fill{height:100%;border-radius:2px;transition:width .5s ease}

/* ── SIDE PANEL ── */
.panel-card .card-header{flex-direction:column;align-items:flex-start;gap:4px}
.panel-section{padding:16px 20px;border-bottom:1px solid var(--border)}
.panel-section:last-child{border-bottom:none}
.panel-section-title{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}

.stat-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0}
.stat-label{font-size:13px;color:var(--muted)}
.stat-value{font-size:13px;font-weight:600;font-family:var(--mono)}

.user-item{
  display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 0;
  border-bottom:1px solid rgba(37,42,66,.5);
}
.user-item:last-child{border-bottom:none}
.user-info{display:flex;align-items:center;gap:10px}
.avatar{
  width:32px;height:32px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent2),var(--accent));
  display:flex;align-items:center;justify-content:center;font-size:13px;
  font-weight:700;color:#fff;flex-shrink:0;
}
.user-name{font-size:13px;font-weight:500}
.user-id{font-size:11px;color:var(--muted);font-family:var(--mono)}
.del-btn{
  background:none;border:none;color:var(--muted);cursor:pointer;
  padding:4px;border-radius:4px;transition:all .2s;
}
.del-btn:hover{color:var(--danger);background:rgba(255,80,80,.1)}

/* ── EMPTY STATE ── */
.empty-state{
  text-align:center;padding:60px 20px;color:var(--muted);
}
.empty-state .icon{font-size:40px;margin-bottom:12px;opacity:.5}
.empty-state p{font-size:14px}

/* ── FOOTER ── */
footer{
  text-align:center;padding:20px;
  color:var(--muted);font-size:12px;
  border-top:1px solid var(--border);
}

/* ── GLOW EFFECTS ── */
.glow-accent{box-shadow:0 0 0 1px rgba(99,212,255,.15),0 4px 24px rgba(99,212,255,.06)}
.glow-purple{box-shadow:0 0 0 1px rgba(130,80,255,.15),0 4px 24px rgba(130,80,255,.06)}

/* ── ANIMATIONS ── */
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.kpi-card,.table-card,.panel-card{animation:fadeIn .35s ease both}
.kpi-card:nth-child(1){animation-delay:.05s}
.kpi-card:nth-child(2){animation-delay:.10s}
.kpi-card:nth-child(3){animation-delay:.15s}
.kpi-card:nth-child(4){animation-delay:.20s}
.table-card{animation-delay:.25s}
.panel-card{animation-delay:.30s}

/* ── RESPONSIVE ── */
@media(max-width:1100px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:820px){.bottom-grid{grid-template-columns:1fr}.kpi-grid{grid-template-columns:1fr 1fr}}
@media(max-width:500px){.kpi-grid{grid-template-columns:1fr}header{padding:0 16px}main{padding:16px}}
</style>
</head>
<body>
<div class="app">

<!-- ── HEADER ── -->
<header>
  <div class="logo">
    <div class="logo-icon">🎓</div>
    <div class="logo-text">
      <span>Smart Attendance</span>
      <small>Face Recognition · Real-Time Tracking</small>
    </div>
  </div>
  <div class="header-right">
    <div class="live-badge"><div class="live-dot"></div>Live</div>
    <div class="time-display" id="clock">--:--:--</div>
    <button class="refresh-btn" style="background:var(--accent2);color:#fff;font-weight:600;border:none;" onclick="openHistoryModal()">📊 4-Month History</button>
    <button class="refresh-btn" style="background:var(--accent3);color:#000;font-weight:600;border:none;" onclick="openUsersModal()">👥 Registered Users</button>
    <button class="refresh-btn" style="background:var(--accent);color:#000;font-weight:600;border:none;" onclick="openRegisterModal()">+ Add Face</button>
    <button class="refresh-btn" onclick="fetchData()">⟳ Refresh</button>
  </div>
</header>

<!-- ── REGISTER MODAL ── -->
<div id="regModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:999;align-items:center;justify-content:center;backdrop-filter:blur(5px);">
  <div style="background:var(--card);padding:28px;border-radius:20px;width:400px;border:1px solid var(--border);box-shadow:0 10px 40px rgba(0,0,0,0.5);">
    <h3 style="margin-bottom:16px;font-size:18px;">📸 Register New Face</h3>
    <video id="regVideo" width="340" height="255" autoplay playsinline style="border-radius:12px;background:#000;margin-bottom:16px;object-fit:cover;border:1px solid var(--border);transform: scaleX(-1);"></video>
    <input id="regName" type="text" placeholder="Enter person's full name" style="width:100%;padding:12px;margin-bottom:20px;border-radius:8px;border:1px solid var(--border);background:var(--bg2);color:var(--text);font-family:var(--font);font-size:14px;outline:none;"/>
    <div style="display:flex;gap:12px;">
       <button onclick="closeRegisterModal()" style="flex:1;padding:12px;border-radius:8px;background:var(--card2);color:var(--text);border:1px solid var(--border);cursor:pointer;font-weight:500;">Cancel</button>
       <button onclick="submitRegistration()" style="flex:2;padding:12px;border-radius:8px;background:var(--accent);color:#000;border:none;cursor:pointer;font-weight:600;box-shadow:0 4px 12px rgba(99,212,255,0.2);">Capture &amp; Save</button>
    </div>
    <div id="regStatus" style="margin-top:16px;font-size:13px;color:var(--muted);text-align:center;"></div>
  </div>
</div>

<!-- ── ATTENDANCE MODAL ── -->
<div id="attendModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:999;align-items:center;justify-content:center;backdrop-filter:blur(5px);">
  <div style="background:var(--card);padding:28px;border-radius:20px;width:400px;border:1px solid var(--border);box-shadow:0 10px 40px rgba(0,0,0,0.5);">
    <h3 style="margin-bottom:16px;font-size:18px;">✓ Mark Attendance</h3>
    <video id="attendVideo" width="340" height="255" autoplay playsinline style="border-radius:12px;background:#000;margin-bottom:16px;object-fit:cover;border:1px solid var(--border);transform: scaleX(-1);"></video>
    <div style="display:flex;gap:12px;">
       <button onclick="closeAttendModal()" style="flex:1;padding:12px;border-radius:8px;background:var(--card2);color:var(--text);border:1px solid var(--border);cursor:pointer;font-weight:500;">Cancel</button>
       <button onclick="submitAttendance()" style="flex:2;padding:12px;border-radius:8px;background:var(--success);color:#000;border:none;cursor:pointer;font-weight:600;box-shadow:0 4px 12px rgba(60,214,120,0.2);">Scan Face</button>
    </div>
    <div id="attendStatus" style="margin-top:16px;font-size:13px;color:var(--muted);text-align:center;"></div>
  </div>
</div>

<!-- ── HISTORY MODAL ── -->
<div id="historyModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:999;align-items:center;justify-content:center;backdrop-filter:blur(5px);">
  <div style="background:var(--card);padding:28px;border-radius:20px;width:500px;max-height:80vh;border:1px solid var(--border);box-shadow:0 10px 40px rgba(0,0,0,0.5);display:flex;flex-direction:column;">
    <h3 style="margin-bottom:16px;font-size:18px;display:flex;justify-content:space-between;align-items:center;">
        <span>📊 4-Month Attendance Summary</span>
        <button onclick="closeHistoryModal()" style="background:none;border:none;color:var(--muted);font-size:24px;cursor:pointer;">&times;</button>
    </h3>
    <input type="text" id="historySearch" placeholder="🔍 Search person by name..." oninput="filterHistory()" style="width:100%;padding:10px 14px;margin-bottom:16px;border-radius:10px;border:1px solid var(--border);background:var(--bg2);color:var(--text);font-family:var(--font);font-size:13px;outline:none;"/>
    <div style="overflow-y:auto;flex:1;margin-bottom:16px;">
        <table style="width:100%">
            <thead>
                <tr>
                    <th style="text-align:left;padding:8px;font-size:12px;color:var(--muted);text-transform:uppercase;">Person Name</th>
                    <th style="text-align:right;padding:8px;font-size:12px;color:var(--muted);text-transform:uppercase;">Days Attended</th>
                </tr>
            </thead>
            <tbody id="historyBody">
                <!-- Data will be injected here -->
            </tbody>
        </table>
    </div>
    <button onclick="closeHistoryModal()" style="padding:12px;border-radius:8px;background:var(--card2);color:var(--text);border:1px solid var(--border);cursor:pointer;font-weight:500;">Close</button>
  </div>
</div>

<!-- ── USERS MODAL ── -->
<div id="usersModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:999;align-items:center;justify-content:center;backdrop-filter:blur(5px);">
  <div style="background:var(--card);padding:28px;border-radius:20px;width:450px;max-height:80vh;border:1px solid var(--border);box-shadow:0 10px 40px rgba(0,0,0,0.5);display:flex;flex-direction:column;">
    <h3 style="margin-bottom:16px;font-size:18px;display:flex;justify-content:space-between;align-items:center;">
        <span>👥 Registered Users</span>
        <button onclick="closeUsersModal()" style="background:none;border:none;color:var(--muted);font-size:24px;cursor:pointer;">&times;</button>
    </h3>
    <div id="usersModalList" style="overflow-y:auto;flex:1;margin-bottom:16px;">
        <!-- Users will be injected here -->
    </div>
    <button onclick="closeUsersModal()" style="padding:12px;border-radius:8px;background:var(--card2);color:var(--text);border:1px solid var(--border);cursor:pointer;font-weight:500;">Close</button>
  </div>
</div>

<!-- ── MAIN ── -->
<main>

  <!-- KPI Cards -->
  <div class="kpi-grid" id="kpiGrid">
    <div class="kpi-card glow-accent">
      <div class="kpi-icon">👥</div>
      <div class="kpi-label">Present Today</div>
      <div class="kpi-value accent" id="kpiToday">–</div>
      <div class="kpi-sub">Marked attendance</div>
    </div>
    <div class="kpi-card glow-purple">
      <div class="kpi-icon">📋</div>
      <div class="kpi-label">All-Time Logs</div>
      <div class="kpi-value accent2" id="kpiAll">–</div>
      <div class="kpi-sub">Total entries</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🧑‍💼</div>
      <div class="kpi-label">Registered Faces</div>
      <div class="kpi-value accent3" id="kpiReg">–</div>
      <div class="kpi-sub">In database</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🕐</div>
      <div class="kpi-label">Last Saved</div>
      <div class="kpi-value warn" id="kpiSaved" style="font-size:20px;padding-top:10px">–</div>
      <div class="kpi-sub">Auto-save time</div>
    </div>
  </div>

  <!-- Bottom Grid -->
  <div class="bottom-grid">

    <!-- Attendance Table -->
    <div class="table-card">
      <div class="card-header">
        <div style="display:flex;align-items:center;gap:12px;">
            <div class="card-title">📅 Today's Attendance</div>
            <span class="badge green" id="presentCount">0 present</span>
        </div>
        <input type="text" id="tableSearch" placeholder="🔍 Quick search..." oninput="fetchData()" style="width:200px;padding:6px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg2);color:var(--text);font-size:12px;outline:none;"/>
      </div>
      <div id="tableWrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>First Seen</th>
              <th>Last Seen</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>Method</th>
              <th>Entries</th>
            </tr>
          </thead>
          <tbody id="attendanceBody">
            <tr><td colspan="8"><div class="empty-state"><div class="icon">📷</div><p>No attendance data yet. Start the main system first.</p></div></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Side Panel -->
    <div class="panel-card">
      <div class="card-header">
        <div class="card-title">📊 System Info</div>
      </div>

      <!-- Stats -->
      <div class="panel-section">
        <div class="panel-section-title">Statistics</div>
        <div class="stat-row"><span class="stat-label">Total Today</span><span class="stat-value" id="sToday">–</span></div>
        <div class="stat-row"><span class="stat-label">All Time</span><span class="stat-value" id="sAll">–</span></div>
        <div class="stat-row"><span class="stat-label">Registered</span><span class="stat-value" id="sReg">–</span></div>
        <div class="stat-row"><span class="stat-label">Last Save</span><span class="stat-value" id="sSave">–</span></div>
      </div>


      <!-- Method Breakdown -->
      <div class="panel-section">
        <div class="panel-section-title">Method Breakdown</div>
        <div id="methodBreakdown">
          <div class="stat-row"><span class="stat-label">Auto (camera)</span><span class="stat-value" id="mAuto">–</span></div>
          <div class="stat-row"><span class="stat-label">Gesture (wave)</span><span class="stat-value" id="mGesture">–</span></div>
        </div>
      </div>
    </div>

  </div>
</main>

<footer>Smart Attendance System &copy; 2025 · Powered by OpenCV &amp; Face Recognition</footer>
</div>

<script>
// ── Clock ──────────────────────────────────────────────────────────────────
function updateClock(){
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toTimeString().slice(0,8);
}
setInterval(updateClock,1000);
updateClock();

// ── Fetch data ─────────────────────────────────────────────────────────────
async function fetchData(){
  try{
    const r = await fetch('/api/data');
    if(!r.ok) throw new Error('Network error');
    const d = await r.json();
    renderKPIs(d.stats, d.registered_count);
    renderTable(d.records);
    renderSidePanel(d.stats, d.registered_count, d.users, d.records);
  } catch(e){
    console.warn('Could not fetch data:', e);
  }
}

// ── KPIs ───────────────────────────────────────────────────────────────────
function renderKPIs(stats, regCount){
  document.getElementById('kpiToday').textContent = stats.total_today ?? '–';
  document.getElementById('kpiAll').textContent   = stats.total_all_time ?? '–';
  document.getElementById('kpiReg').textContent   = regCount ?? '–';
  document.getElementById('kpiSaved').textContent = stats.last_save ?? '–';
}

// ── Table ──────────────────────────────────────────────────────────────────
function renderTable(records){
  const tbody = document.getElementById('attendanceBody');
  const badge = document.getElementById('presentCount');
  const search = document.getElementById('tableSearch').value.toLowerCase();
  
  let entries = Object.entries(records || {});
  
  // Filter by search
  if(search) {
    entries = entries.filter(([name]) => name.toLowerCase().includes(search));
  }
  
  badge.textContent = entries.length + (search ? ' matching' : ' present');

  if(entries.length === 0){
    tbody.innerHTML = `<tr><td colspan="8"><div class="empty-state"><div class="icon">${search ? '🔍' : '📷'}</div><p>${search ? 'No matches found.' : 'No attendance data yet.'}</p></div></td></tr>`;
    return;
  }

  tbody.innerHTML = entries.map(([name,r],i)=>{
    const conf = (r.confidence * 100).toFixed(0);
    const fillColor = r.confidence > 0.7 ? 'var(--success)' : r.confidence > 0.4 ? 'var(--warning)' : 'var(--danger)';
    return `<tr>
      <td style="color:var(--muted);font-family:var(--mono)">${i+1}</td>
      <td style="font-weight:600">${esc(name)}</td>
      <td style="font-family:var(--mono);font-size:12px">${esc(r.first_seen)}</td>
      <td style="font-family:var(--mono);font-size:12px">${esc(r.last_seen)}</td>
      <td><span class="status-pill present">✓ ${esc(r.status)}</span></td>
      <td>
        <div style="display:flex;align-items:center;gap:6px">
          <div class="confidence-bar"><div class="confidence-fill" style="width:${conf}%;background:${fillColor}"></div></div>
          <span style="font-size:11px;font-family:var(--mono)">${conf}%</span>
        </div>
      </td>
      <td><span class="method-tag">${esc(r.method)}</span></td>
      <td style="font-family:var(--mono)">${r.entries}</td>
    </tr>`;
  }).join('');
}

// ── Side panel ─────────────────────────────────────────────────────────────
function renderSidePanel(stats, regCount, users, records){
  document.getElementById('sToday').textContent  = stats.total_today ?? '–';
  document.getElementById('sAll').textContent    = stats.total_all_time ?? '–';
  document.getElementById('sReg').textContent    = regCount ?? '–';
  document.getElementById('sSave').textContent   = stats.last_save ?? '–';

  // Cache users for the modal
  window._cachedUsers = users || [];

  // Method breakdown
  const entries = Object.values(records || {});
  const auto    = entries.filter(r=>r.method==='auto' || r.method==='web-scan').length;
  const gesture = entries.filter(r=>r.method==='gesture').length;
  document.getElementById('mAuto').textContent    = auto;
  document.getElementById('mGesture').textContent = gesture;
}

function openUsersModal() {
  document.getElementById('usersModal').style.display = 'flex';
  const list = document.getElementById('usersModalList');
  const users = window._cachedUsers || [];
  
  if(users.length === 0){
    list.innerHTML = '<div class="empty-state" style="padding:20px 0"><div class="icon" style="font-size:24px">🔒</div><p style="font-size:12px">No users registered</p></div>';
  } else {
    list.innerHTML = users.map(([n,id])=>`
      <div class="user-item">
        <div class="user-info">
          <div class="avatar">${esc(n).charAt(0).toUpperCase()}</div>
          <div><div class="user-name">${esc(n)}</div><div class="user-id">${esc(id)}</div></div>
        </div>
        <button class="del-btn" onclick="deleteUser('${esc(n)}')" title="Delete User">🗑️</button>
      </div>`).join('');
  }
}

function closeUsersModal() {
  document.getElementById('usersModal').style.display = 'none';
}

async function deleteUser(name){
  if(!confirm(`Are you sure you want to delete ${name}?`)) return;
  try{
    const res = await fetch('/api/delete_user', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name})
    });
    const d = await res.json();
    if(d.success){
      fetchData();
    } else {
      alert(d.message);
    }
  } catch(e){ console.error(e); }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function esc(s){
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Registration Modal ─────────────────────────────────────────────────────
let stream = null;
async function openRegisterModal() {
  document.getElementById('regModal').style.display = 'flex';
  document.getElementById('regStatus').textContent = 'Connecting to camera...';
  document.getElementById('regName').value = '';
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true});
    document.getElementById('regVideo').srcObject = stream;
    document.getElementById('regStatus').textContent = 'Position your face clearly in the frame.';
  } catch (err) {
    document.getElementById('regStatus').textContent = 'Camera access denied or unavailable.';
  }
}
function closeRegisterModal() {
  document.getElementById('regModal').style.display = 'none';
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
}
async function submitRegistration() {
  const name = document.getElementById('regName').value.trim();
  if (!name) return alert('Please enter a name');
  const video = document.getElementById('regVideo');
  if (!video.videoWidth) return alert('Camera not ready');
  
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; 
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.9);
  
  document.getElementById('regStatus').textContent = 'Registering...';
  document.getElementById('regStatus').style.color = 'var(--accent)';
  
  try {
    const res = await fetch('/api/register', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, image: b64})
    });
    const data = await res.json();
    document.getElementById('regStatus').textContent = data.message;
    document.getElementById('regStatus').style.color = data.success ? 'var(--success)' : 'var(--danger)';
    if(data.success) {
      setTimeout(() => { closeRegisterModal(); fetchData(); }, 2000);
    }
  } catch (e) {
    document.getElementById('regStatus').textContent = 'Error during registration.';
    document.getElementById('regStatus').style.color = 'var(--danger)';
  }
}

// ── Attendance Modal ───────────────────────────────────────────────────────
let attendStream = null;
async function openAttendModal() {
  document.getElementById('attendModal').style.display = 'flex';
  document.getElementById('attendStatus').textContent = 'Connecting to camera...';
  try {
    attendStream = await navigator.mediaDevices.getUserMedia({video: true});
    document.getElementById('attendVideo').srcObject = attendStream;
    document.getElementById('attendStatus').textContent = 'Position your face clearly and click Scan.';
  } catch (err) {
    document.getElementById('attendStatus').textContent = 'Camera access denied or unavailable.';
  }
}
function closeAttendModal() {
  document.getElementById('attendModal').style.display = 'none';
  if (attendStream) { attendStream.getTracks().forEach(t => t.stop()); attendStream = null; }
}
async function submitAttendance() {
  const video = document.getElementById('attendVideo');
  if (!video.videoWidth) return alert('Camera not ready');
  
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; 
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.9);
  
  document.getElementById('attendStatus').textContent = 'Scanning face...';
  document.getElementById('attendStatus').style.color = 'var(--accent)';
  
  try {
    const res = await fetch('/api/attend', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: b64})
    });
    const data = await res.json();
    document.getElementById('attendStatus').textContent = data.message;
    document.getElementById('attendStatus').style.color = data.success ? 'var(--success)' : 'var(--warning)';
    if(data.success) {
      setTimeout(() => { closeAttendModal(); fetchData(); }, 2000);
    }
  } catch (e) {
    document.getElementById('attendStatus').textContent = 'Error during scanning.';
    document.getElementById('attendStatus').style.color = 'var(--danger)';
  }
}

async function openHistoryModal() {
  document.getElementById('historyModal').style.display = 'flex';
  document.getElementById('historySearch').value = '';
  const tbody = document.getElementById('historyBody');
  tbody.innerHTML = '<tr><td colspan="2" style="text-align:center;padding:20px;">Loading history...</td></tr>';
  
  try {
    const res = await fetch('/api/history');
    window._cachedHistory = await res.json();
    renderHistoryTable(window._cachedHistory);
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="2" style="text-align:center;color:var(--danger);padding:20px;">Error loading history.</td></tr>';
  }
}

function renderHistoryTable(data) {
  const tbody = document.getElementById('historyBody');
  const search = document.getElementById('historySearch').value.toLowerCase();
  
  const filtered = data.filter(item => item.name.toLowerCase().includes(search));
  
  if(!filtered || filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="2" style="text-align:center;padding:20px;color:var(--muted);">${search ? 'No names match your search.' : 'No history records found yet.'}</td></tr>`;
  } else {
      tbody.innerHTML = filtered.map(item => `
          <tr>
              <td style="padding:12px;border-bottom:1px solid rgba(255,255,255,0.05);font-weight:600;">${esc(item.name)}</td>
              <td style="padding:12px;border-bottom:1px solid rgba(255,255,255,0.05);text-align:right;font-family:var(--mono);color:var(--accent);font-size:15px;">${item.count} days</td>
          </tr>
      `).join('');
  }
}

function filterHistory() {
    if(window._cachedHistory) renderHistoryTable(window._cachedHistory);
}

function closeHistoryModal() {
  document.getElementById('historyModal').style.display = 'none';
}

// ── Auto-refresh every 5 s ─────────────────────────────────────────────────
fetchData();
setInterval(fetchData, 5000);
</script>
</body>
</html>
"""


# ── API ROUTES ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    """Return all relevant data as JSON."""
    db.load_csv()          # Reload from disk so we always see latest
    stats   = db.get_statistics()
    records = db.get_today_records()
    users   = db.recognizer_users if hasattr(db, 'recognizer_users') else []

    # Serialize records
    serialized = {}
    for name, rec in records.items():
        serialized[name] = {
            "name":       rec["name"],
            "id":         rec.get("id", "N/A"),
            "first_seen": rec["first_seen"].strftime("%H:%M:%S"),
            "last_seen":  rec["last_seen"].strftime("%H:%M:%S"),
            "status":     rec.get("status", "PRESENT"),
            "confidence": rec.get("confidence", 0.0),
            "method":     rec.get("method", "auto"),
            "entries":    rec.get("entries", 1),
        }

    # Try to get registered users from the recognizer database
    registered_users = []
    try:
        from recognizer import FaceRecognizer
        recog = FaceRecognizer()
        registered_users = recog.get_registered_users()
    except Exception:
        pass

    return jsonify({
        "stats":            stats,
        "records":          serialized,
        "registered_count": len(registered_users),
        "users":            registered_users,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    ok1, msg1 = db.save_csv()
    ok2, msg2 = db.save_excel()
    _, path   = db.generate_daily_report()
    return jsonify({"success": ok1 and ok2, "csv": msg1, "excel": msg2, "report": path})


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json
    name = data.get("name", "").strip()
    img_b64 = data.get("image", "")
    if not name or not img_b64:
        return jsonify({"success": False, "message": "Missing name or image"}), 400
    
    try:
        header, encoded = img_b64.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        from recognizer import FaceRecognizer
        recog = FaceRecognizer()
        success, msg = recog.register_face(frame, name)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500


@app.route("/api/delete_user", methods=["POST"])
def api_delete_user():
    data = request.json
    name = data.get("name")
    if not name:
        return jsonify({"success": False, "message": "Name required"}), 400
    
    try:
        from recognizer import FaceRecognizer
        recog = FaceRecognizer()
        success = recog.delete_user(name)
        
        # Also remove from current attendance log if they exist
        if success and name in db.records:
            del db.records[name]
            db.save_csv()
            db.save_excel()
            
        return jsonify({"success": success, "message": "User deleted" if success else "User not found"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/history")
def api_history():
    """Return aggregated 4-month history."""
    try:
        report = db.get_history_report(days=120)
        return jsonify(report)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Attendance Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
