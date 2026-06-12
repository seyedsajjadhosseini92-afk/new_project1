"""
SecureVision Pro — Face Recognition Access Control System
"""

import json
import os
import queue
import shutil
import threading
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter as tk

import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

from core.config import CONFIG
from core.database import db
from core.face_engine import engine
from core.tracker import Tracker

# ── Design tokens ──────────────────────────────────────────────────────────────
C = {
    'bg':       '#060b14',
    'bg2':      '#08101d',
    'panel':    '#091524',
    'card':     '#0d1e3a',
    'card_h':   '#112448',
    'card_sel': '#163060',
    'sidebar':  '#040a14',
    'header':   '#050c1a',
    'border':   '#162840',
    'bdr_a':    '#00c8ee',
    'bdr_g':    '#00e676',
    'bdr_y':    '#ffd60a',
    'bdr_r':    '#ff3d5a',
    'bdr_p':    '#b388ff',
    'accent':   '#00d4ff',
    'accent_d': '#0088cc',
    'teal':     '#00e5a8',
    'green':    '#00e676',
    'green_d':  '#00aa55',
    'red':      '#ff3d5a',
    'red_d':    '#8b0020',
    'yellow':   '#ffd60a',
    'amber':    '#f59e0b',
    'orange':   '#ff7043',
    'purple':   '#b388ff',
    'text':     '#ddeef8',
    'text2':    '#7aa0c8',
    'sub':      '#3a5878',
    'dim':      '#0f2030',
    'sep':      '#0d1c2e',
}

DEPT_COLORS = {
    'IT':  '#00c8ee', 'HR': '#00e676', 'Finance': '#ffd60a',
    'Security': '#ff3d5a', 'Management': '#b388ff', 'Operations': '#f59e0b',
}


# ── Widget helpers ─────────────────────────────────────────────────────────────

def _btn(parent, text, command, bg=None, fg=None, **kw):
    bg = bg or C['accent']
    fg = fg or (C['bg'] if bg == C['accent'] else C['text'])
    return tk.Button(parent, text=text, command=command,
                     bg=bg, fg=fg,
                     activebackground=C['card_sel'], activeforeground=C['text'],
                     relief='flat', cursor='hand2', **kw)


def _sep(parent, orient='h', color=None, **kw):
    color = color or C['sep']
    if orient == 'h':
        return tk.Frame(parent, bg=color, height=1, **kw)
    return tk.Frame(parent, bg=color, width=1, **kw)


def _card(parent, glow=None, **kw):
    glow = glow or C['border']
    outer = tk.Frame(parent, bg=glow, **kw)
    inner = tk.Frame(outer, bg=C['card'])
    inner.pack(fill='both', expand=True, padx=1, pady=1)
    return outer, inner


def _lbl(parent, text, font=None, fg=None, bg=None, **kw):
    return tk.Label(parent, text=text,
                    font=font or ('Arial', 10),
                    fg=fg or C['text'],
                    bg=bg or C['card'], **kw)


def _scrollable(parent, bg=None):
    bg = bg or C['bg']
    canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
    inner = tk.Frame(canvas, bg=bg)
    inner.bind('<Configure>',
               lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.create_window((0, 0), window=inner, anchor='nw')
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    canvas.bind('<Enter>', lambda e: canvas.bind_all(
        '<MouseWheel>', lambda ev: canvas.yview_scroll(-1*(ev.delta//120), 'units')))
    canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))
    return canvas, inner


# ── Main app ───────────────────────────────────────────────────────────────────

class SecurityApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('SecureVision Pro  —  Face Recognition Access Control')
        self.root.geometry('1520x900')
        self.root.minsize(1280, 760)
        self.root.configure(bg=C['bg'])

        self.tracker                   = Tracker()
        self._frame_q                  = queue.Queue(maxsize=2)
        self._cam_running              = False
        self._cap                      = None
        self._res_idx                  = CONFIG.default_resolution
        self._fps                      = 0.0
        self._fps_cnt                  = 0
        self._fps_ts                   = datetime.now()
        self._sel_pid: str | None      = None
        self._photo_refs: list         = []
        self._grid_refs:  list         = []
        self._wl_refs:    list         = []
        self._name_cache: dict         = {}
        self._dept_cache: dict         = {}
        self._last_stats: dict         = {}
        self._last_active_pids: set    = set()
        self._current_page             = 'dashboard'
        self._page_frames: dict        = {}
        self._nav_btns:    dict        = {}
        self._merge_mode               = False
        self._merge_selected: list     = []
        self._watchlist_embs: list     = []
        self._alarm_info: dict | None  = None
        self._alarm_active             = False
        self._alarm_last_seen: datetime | None = None
        self._last_alarm_time: datetime | None = None

        # Dashboard KPI vars
        self._kpi_total    = tk.StringVar(value='—')
        self._kpi_inside   = tk.StringVar(value='—')
        self._kpi_entries  = tk.StringVar(value='—')
        self._kpi_exits    = tk.StringVar(value='—')
        self._kpi_live     = tk.StringVar(value='—')

        self._setup_styles()
        self._build_ui()
        self._load_watchlist_embs()
        self._refresh_caches()
        self._start_camera()
        self._schedule_loops()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('D.Treeview',
                    background='#0d1e3a', foreground=C['text'],
                    fieldbackground='#0d1e3a',
                    font=('Consolas', 10), rowheight=26)
        s.configure('D.Treeview.Heading',
                    background=C['panel'], foreground=C['accent'],
                    font=('Arial', 10, 'bold'))
        s.map('D.Treeview',
              background=[('selected', C['card_sel'])],
              foreground=[('selected', C['text'])])
        s.configure('TNotebook', background=C['panel'], borderwidth=0)
        s.configure('TNotebook.Tab', background=C['card'], foreground=C['sub'],
                    padding=[14, 6], font=('Arial', 10))
        s.map('TNotebook.Tab',
              background=[('selected', C['card_sel'])],
              foreground=[('selected', C['accent'])])
        s.configure('Vertical.TScrollbar',
                    background=C['panel'], troughcolor=C['bg'],
                    arrowcolor=C['sub'], borderwidth=0)
        s.configure('Horizontal.TScale', troughcolor=C['dim'], background=C['accent'])

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        _sep(self.root, color=C['bdr_a']).pack(fill='x')

        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(fill='both', expand=True)

        self._build_sidebar(body)
        _sep(body, orient='v', color=C['border']).pack(side='left', fill='y')

        area = tk.Frame(body, bg=C['bg'])
        area.pack(side='left', fill='both', expand=True)

        self._build_dashboard_page(area)
        self._build_live_page(area)
        self._build_employees_page(area)
        self._build_attendance_page(area)
        self._build_recognition_page(area)
        self._build_watchlist_page(area)
        self._build_settings_page(area)

        self._switch_page('dashboard')

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        bar = tk.Frame(self.root, bg=C['header'], height=64)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        # Logo + brand
        brand = tk.Frame(bar, bg=C['header'])
        brand.pack(side='left', padx=18, pady=10)
        tk.Label(brand, text='◈', font=('Arial', 26), bg=C['header'],
                 fg=C['accent']).pack(side='left', padx=(0, 10))
        titles = tk.Frame(brand, bg=C['header'])
        titles.pack(side='left')
        tk.Label(titles, text='SecureVision Pro',
                 font=('Arial', 15, 'bold'), bg=C['header'],
                 fg=C['text']).pack(anchor='w')
        tk.Label(titles, text='Face Recognition Access Control',
                 font=('Arial', 8), bg=C['header'],
                 fg=C['sub']).pack(anchor='w')

        _sep(bar, orient='v', color=C['border']).pack(side='left', fill='y', pady=10)

        # Right side badges
        right = tk.Frame(bar, bg=C['header'])
        right.pack(side='right', padx=12)

        self._alarm_lbl = tk.Label(right, text='', font=('Arial', 11, 'bold'),
                                   bg=C['header'], fg=C['red'])
        self._alarm_lbl.pack(side='right', padx=12)

        _sep(right, orient='v', color=C['border']).pack(side='right', fill='y', pady=10)

        # FPS
        fps_f = tk.Frame(right, bg=C['dim'], padx=12, pady=4)
        fps_f.pack(side='right', padx=6)
        tk.Label(fps_f, text='FPS', font=('Arial', 7, 'bold'),
                 bg=C['dim'], fg=C['sub']).pack()
        self._lbl_fps = tk.Label(fps_f, text='--', font=('Consolas', 14, 'bold'),
                                  bg=C['dim'], fg=C['yellow'])
        self._lbl_fps.pack()

        _sep(right, orient='v', color=C['border']).pack(side='right', fill='y', pady=10)

        # Active
        act_f = tk.Frame(right, bg=C['dim'], padx=12, pady=4)
        act_f.pack(side='right', padx=6)
        tk.Label(act_f, text='LIVE', font=('Arial', 7, 'bold'),
                 bg=C['dim'], fg=C['sub']).pack()
        self._lbl_active = tk.Label(act_f, text='0', font=('Consolas', 14, 'bold'),
                                     bg=C['dim'], fg=C['green'])
        self._lbl_active.pack()

        _sep(right, orient='v', color=C['border']).pack(side='right', fill='y', pady=10)

        # Clock + date
        clock_f = tk.Frame(right, bg=C['header'], padx=14)
        clock_f.pack(side='right')
        self._lbl_clock = tk.Label(clock_f, text='--:--:--',
                                    font=('Consolas', 18, 'bold'),
                                    bg=C['header'], fg=C['accent'])
        self._lbl_clock.pack()
        self._lbl_date = tk.Label(clock_f, text='',
                                   font=('Arial', 8), bg=C['header'], fg=C['sub'])
        self._lbl_date.pack()
        self._tick_clock()

    def _tick_clock(self):
        now = datetime.now()
        self._lbl_clock.config(text=now.strftime('%H:%M:%S'))
        self._lbl_date.config(text=now.strftime('%A  %Y-%m-%d'))
        self.root.after(1000, self._tick_clock)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C['sidebar'], width=150)
        sb.pack(side='left', fill='y')
        sb.pack_propagate(False)

        tk.Label(sb, text='NAVIGATION', font=('Arial', 7, 'bold'),
                 bg=C['sidebar'], fg=C['sub']).pack(pady=(16, 8))

        nav_items = [
            ('dashboard',   '⬡', 'Dashboard'),
            ('live',        '📹', 'Live Feed'),
            ('employees',   '👤', 'Employees'),
            ('attendance',  '📋', 'Attendance'),
            ('recognition', '🔍', 'Recognition'),
            ('watchlist',   '🔔', 'Watchlist'),
            ('settings',    '⚙', 'Settings'),
        ]
        for page_id, icon, label in nav_items:
            self._make_nav_btn(sb, page_id, icon, label)

        _sep(sb, color=C['border']).pack(fill='x', pady=12, padx=10)
        tk.Label(sb, text='v2.0', font=('Arial', 8),
                 bg=C['sidebar'], fg=C['sub']).pack()

    def _make_nav_btn(self, parent, page_id, icon, label):
        wrap = tk.Frame(parent, bg=C['sidebar'])
        wrap.pack(fill='x', pady=1)
        indicator = tk.Frame(wrap, bg=C['sidebar'], width=3)
        indicator.pack(side='left', fill='y')
        btn = tk.Button(wrap,
                        text=f'{icon}  {label}',
                        font=('Arial', 10), anchor='w', padx=12, pady=10,
                        bg=C['sidebar'], fg=C['text2'],
                        activebackground=C['card'], activeforeground=C['accent'],
                        relief='flat', cursor='hand2',
                        command=lambda p=page_id: self._switch_page(p))
        btn.pack(side='left', fill='x', expand=True)
        self._nav_btns[page_id] = (wrap, btn, indicator)

    def _switch_page(self, page_id: str):
        self._current_page = page_id
        for pid, frame in self._page_frames.items():
            if pid == page_id:
                frame.pack(fill='both', expand=True)
            else:
                frame.pack_forget()
        for pid, (wrap, btn, ind) in self._nav_btns.items():
            if pid == page_id:
                wrap.config(bg=C['card'])
                btn.config(bg=C['card'], fg=C['accent'])
                ind.config(bg=C['accent'])
            else:
                wrap.config(bg=C['sidebar'])
                btn.config(bg=C['sidebar'], fg=C['text2'])
                ind.config(bg=C['sidebar'])
        if page_id == 'employees':
            self.root.after(80, self._refresh_employees_grid)
        elif page_id == 'attendance':
            self.root.after(80, self._refresh_attendance)
        elif page_id == 'watchlist':
            self._refresh_watchlist_ui()
        elif page_id == 'dashboard':
            self.root.after(80, self._refresh_dashboard)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — Dashboard
    # ══════════════════════════════════════════════════════════════════════════

    def _build_dashboard_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['dashboard'] = frame

        # Page header
        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='  ⬡  DASHBOARD  —  Overview & Live Stats',
                 font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['text']).pack(side='left', pady=14)
        _btn(hdr, '↻  Refresh', self._refresh_dashboard,
             bg=C['dim'], fg=C['text2'], font=('Arial', 9), padx=12).pack(side='right', padx=12, pady=12)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        # KPI row
        kpi_row = tk.Frame(frame, bg=C['bg'])
        kpi_row.pack(fill='x', padx=16, pady=14)

        kpis = [
            ('👤', 'REGISTERED',    self._kpi_total,   C['bdr_a']),
            ('✅', 'INSIDE NOW',     self._kpi_inside,  C['bdr_g']),
            ('📥', 'ENTRIES TODAY',  self._kpi_entries, C['bdr_y']),
            ('📤', 'EXITS TODAY',    self._kpi_exits,   C['orange']),
            ('👁', 'LIVE DETECTED',  self._kpi_live,    C['bdr_p']),
        ]
        for icon, title, var, color in kpis:
            self._kpi_card(kpi_row, icon, title, var, color)

        _sep(frame, color=C['border']).pack(fill='x', padx=16)

        # Main body: attendance quick-view + activity feed
        body = tk.Frame(frame, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=16, pady=10)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # Left: who's inside right now
        left_outer, left = _card(body, glow=C['bdr_g'])
        left_outer.grid(row=0, column=0, sticky='nsew', padx=(0, 8))

        lhdr = tk.Frame(left, bg=C['card'])
        lhdr.pack(fill='x', padx=14, pady=(12, 6))
        tk.Label(lhdr, text='✅  Currently Inside',
                 font=('Arial', 11, 'bold'), bg=C['card'], fg=C['green']).pack(side='left')
        self._dash_inside_count = tk.Label(lhdr, text='',
                                            font=('Consolas', 10), bg=C['card'], fg=C['text2'])
        self._dash_inside_count.pack(side='right')

        _sep(left, color=C['border']).pack(fill='x')

        cols = ('Name', 'Department', 'Entry Time')
        self._dash_inside_tree = ttk.Treeview(left, columns=cols, show='headings',
                                               style='D.Treeview', height=12)
        for c, w in zip(cols, [200, 140, 120]):
            self._dash_inside_tree.heading(c, text=c)
            self._dash_inside_tree.column(c, width=w)
        sb2 = ttk.Scrollbar(left, orient='vertical',
                             command=self._dash_inside_tree.yview)
        self._dash_inside_tree.configure(yscrollcommand=sb2.set)
        sb2.pack(side='right', fill='y', padx=(0, 6))
        self._dash_inside_tree.pack(fill='both', expand=True, padx=8, pady=6)

        # Right: recent activity
        right_outer, right = _card(body, glow=C['bdr_a'])
        right_outer.grid(row=0, column=1, sticky='nsew')

        rhdr = tk.Frame(right, bg=C['card'])
        rhdr.pack(fill='x', padx=14, pady=(12, 6))
        tk.Label(rhdr, text='⚡  Recent Activity',
                 font=('Arial', 11, 'bold'), bg=C['card'], fg=C['accent']).pack(side='left')

        _sep(right, color=C['border']).pack(fill='x')

        self._activity_frame = tk.Frame(right, bg=C['card'])
        self._activity_frame.pack(fill='both', expand=True, padx=10, pady=8)

    def _kpi_card(self, parent, icon, title, var, color):
        outer = tk.Frame(parent, bg=color)
        outer.pack(side='left', fill='both', expand=True, padx=5)
        inner = tk.Frame(outer, bg=C['card'])
        inner.pack(fill='both', expand=True, padx=2, pady=2)
        top = tk.Frame(inner, bg=C['card'])
        top.pack(fill='x', padx=14, pady=(12, 2))
        tk.Label(top, text=icon, font=('Arial', 16), bg=C['card'], fg=color).pack(side='left')
        tk.Label(top, text=title, font=('Arial', 8, 'bold'),
                 bg=C['card'], fg=C['sub']).pack(side='left', padx=8)
        tk.Label(inner, textvariable=var, font=('Consolas', 30, 'bold'),
                 bg=C['card'], fg=color).pack(padx=14, pady=(2, 12))

    def _refresh_dashboard(self):
        counts = db.get_today_counts()
        self._kpi_total.set(str(counts['total']))
        self._kpi_entries.set(str(counts['entries']))
        self._kpi_exits.set(str(counts['exits']))
        self._kpi_live.set(str(len(self.tracker.active_pids)))

        inside = db.get_currently_inside()
        self._kpi_inside.set(str(len(inside)))
        self._dash_inside_count.config(text=f'{len(inside)} people')

        for row in self._dash_inside_tree.get_children():
            self._dash_inside_tree.delete(row)
        for row in inside:
            name = row['name'] or f'Person {row["id"]}'
            dept = row['department'] or '—'
            t    = (row['entry_time'] or '')[-8:][:5]
            self._dash_inside_tree.insert('', 'end', values=(name, dept, t))

        # Activity feed
        for w in self._activity_frame.winfo_children():
            w.destroy()
        for act in db.get_recent_activity(14):
            name   = act['name'] or f'ID {act["person_id"]}'
            action = act['action']
            t      = (act['time'] or '')[-8:][:5]
            color  = C['green'] if action == 'ENTRY' else C['red']
            icon   = '→' if action == 'ENTRY' else '←'
            row_f  = tk.Frame(self._activity_frame, bg=C['card'])
            row_f.pack(fill='x', pady=1)
            tk.Frame(row_f, bg=color, width=3).pack(side='left', fill='y')
            tk.Label(row_f, text=f' {icon} ', font=('Arial', 11, 'bold'),
                     bg=C['card'], fg=color).pack(side='left')
            tk.Label(row_f, text=name, font=('Arial', 10),
                     bg=C['card'], fg=C['text']).pack(side='left')
            tk.Label(row_f, text=t, font=('Consolas', 9),
                     bg=C['card'], fg=C['sub']).pack(side='right', padx=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — Live Feed
    # ══════════════════════════════════════════════════════════════════════════

    def _build_live_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['live'] = frame
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        # Left panel: active persons
        left = tk.Frame(frame, bg=C['panel'], width=280)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.grid_propagate(False)

        lhdr = tk.Frame(left, bg=C['panel'], height=42)
        lhdr.pack(fill='x')
        lhdr.pack_propagate(False)
        tk.Label(lhdr, text='  👤  DETECTED PERSONS',
                 font=('Arial', 9, 'bold'), bg=C['panel'], fg=C['sub']).pack(side='left', pady=10)
        _btn(lhdr, '↻', self._refresh_caches,
             bg=C['dim'], fg=C['text2'], font=('Arial', 11), padx=6, pady=2).pack(
             side='right', padx=6, pady=6)
        _sep(left, color=C['border']).pack(fill='x')

        _, self._live_list_inner = _scrollable(left, bg=C['panel'])

        # Center: camera
        center = tk.Frame(frame, bg=C['bg'])
        center.grid(row=0, column=1, sticky='nsew', padx=6)
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)

        cam_wrap = tk.Frame(center, bg=C['bdr_a'])
        cam_wrap.grid(row=0, column=0, sticky='nsew', pady=(0, 6))
        self._cam_canvas = tk.Canvas(cam_wrap, bg='#010408', highlightthickness=0)
        self._cam_canvas.pack(fill='both', expand=True, padx=1, pady=1)

        ctrl = tk.Frame(center, bg=C['panel'], height=40)
        ctrl.grid(row=1, column=0, sticky='ew')
        self._lbl_res = tk.Label(ctrl, text='640×480',
                                  font=('Consolas', 10), bg=C['panel'], fg=C['text2'])
        self._lbl_res.pack(side='left', padx=14)
        _sep(ctrl, orient='v').pack(side='left', fill='y', pady=8)
        tk.Label(ctrl, text='Resolution:', font=('Arial', 9),
                 bg=C['panel'], fg=C['sub']).pack(side='left', padx=8)
        _btn(ctrl, '▲ Quality', self._res_up,
             bg=C['dim'], fg=C['text2'], font=('Arial', 9), padx=10).pack(side='right', padx=4, pady=5)
        _btn(ctrl, '▼ Quality', self._res_down,
             bg=C['dim'], fg=C['text2'], font=('Arial', 9), padx=10).pack(side='right', padx=4, pady=5)

    def _refresh_live_list(self):
        if not hasattr(self, '_live_list_inner'):
            return
        for w in self._live_list_inner.winfo_children():
            w.destroy()
        active = self.tracker.active_pids
        if not active:
            tk.Label(self._live_list_inner,
                     text='No faces detected', font=('Arial', 10),
                     bg=C['panel'], fg=C['sub']).pack(pady=30)
            return
        for pid in active:
            name = self._name_cache.get(pid, f'Person {pid}')
            dept = self._dept_cache.get(pid, '')
            card = tk.Frame(self._live_list_inner, bg=C['card_sel'], cursor='hand2')
            card.pack(fill='x', padx=6, pady=2)
            tk.Frame(card, bg=C['green'], width=3).pack(side='left', fill='y')
            photo_path = os.path.join(CONFIG.people_dir, pid, 'photo.jpg')
            if os.path.exists(photo_path):
                try:
                    img = Image.open(photo_path).resize((44, 44))
                    ph  = ImageTk.PhotoImage(img)
                    self._photo_refs.append(ph)
                    tk.Label(card, image=ph, bg=C['card_sel']).pack(side='left', padx=6, pady=4)
                except Exception:
                    pass
            info = tk.Frame(card, bg=C['card_sel'])
            info.pack(side='left', fill='x', expand=True, pady=4)
            tk.Label(info, text=name, font=('Arial', 10, 'bold'),
                     bg=C['card_sel'], fg=C['text'], anchor='w').pack(fill='x')
            if dept:
                tk.Label(info, text=dept, font=('Arial', 8),
                         bg=C['card_sel'], fg=C['text2'], anchor='w').pack(fill='x')
            tk.Frame(card, bg=C['green'], width=8, height=8).pack(
                side='right', padx=10)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3 — Employees
    # ══════════════════════════════════════════════════════════════════════════

    def _build_employees_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['employees'] = frame

        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='  👤  EMPLOYEE MANAGEMENT',
                 font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['text']).pack(side='left', pady=14)
        _btn(hdr, '↻  Refresh', self._refresh_employees_grid,
             bg=C['dim'], fg=C['text2'], font=('Arial', 9), padx=12).pack(side='right', padx=6, pady=10)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        # Toolbar: search + dept filter + merge
        toolbar = tk.Frame(frame, bg=C['panel'])
        toolbar.pack(fill='x', padx=12, pady=8)

        self._emp_search_var = tk.StringVar()
        self._emp_search_var.trace_add('write', lambda *_: self._refresh_employees_grid())
        search_entry = tk.Entry(toolbar, textvariable=self._emp_search_var,
                                bg=C['dim'], fg=C['text'], insertbackground=C['accent'],
                                relief='flat', font=('Arial', 11),
                                highlightthickness=1, highlightbackground=C['border'],
                                highlightcolor=C['accent'])
        search_entry.pack(side='left', ipady=6, padx=(0, 10), ipadx=8)
        search_entry.insert(0, '🔍  Search employees…')
        search_entry.bind('<FocusIn>',  lambda e: search_entry.delete(0, 'end')
                         if '🔍' in search_entry.get() else None)
        search_entry.bind('<FocusOut>', lambda e: (search_entry.insert(0, '🔍  Search employees…')
                         if not search_entry.get() else None))

        self._btn_emp_merge = _btn(toolbar, '⛙  Merge Mode',
                                    self._toggle_merge_mode,
                                    bg=C['dim'], fg=C['text2'],
                                    font=('Arial', 9), padx=12)
        self._btn_emp_merge.pack(side='right', padx=4)

        self._btn_do_merge = _btn(toolbar, '⛙  Merge Selected',
                                   self._do_manual_merge,
                                   bg=C['amber'], fg=C['bg'],
                                   font=('Arial', 9), padx=12)

        self._lbl_emp_status = tk.Label(toolbar, text='', font=('Arial', 9),
                                         bg=C['panel'], fg=C['green'])
        self._lbl_emp_status.pack(side='right', padx=12)

        _sep(frame, color=C['border']).pack(fill='x')

        _, self._emp_grid_inner = _scrollable(frame, bg=C['bg'])

    def _refresh_employees_grid(self):
        if not hasattr(self, '_emp_grid_inner'):
            return
        for w in self._emp_grid_inner.winfo_children():
            w.destroy()
        self._grid_refs = []

        q = self._emp_search_var.get().lower()
        if '🔍' in q:
            q = ''

        rows = db.get_all_people()
        COLS = 6
        shown = 0
        for idx, row in enumerate(rows):
            pid  = row['id']
            name = row['name'] or f'Person {pid}'
            dept = row['department'] or ''
            if q and q not in name.lower() and q not in (dept or '').lower() and q not in pid:
                continue
            r, c    = divmod(shown, COLS)
            shown  += 1
            sel     = pid in self._merge_selected
            is_live = pid in self.tracker.active_pids
            border_c = C['amber'] if sel else (C['bdr_g'] if is_live else C['border'])
            card_bg  = C['card_h'] if sel else C['card']

            outer = tk.Frame(self._emp_grid_inner, bg=border_c,
                             width=148, height=190, cursor='hand2')
            outer.grid(row=r, column=c, padx=7, pady=7, sticky='n')
            outer.grid_propagate(False)
            card = tk.Frame(outer, bg=card_bg)
            card.pack(fill='both', expand=True, padx=1, pady=1)

            photo_path = os.path.join(CONFIG.people_dir, pid, 'photo.jpg')
            if os.path.exists(photo_path):
                try:
                    img  = Image.open(photo_path).resize((124, 124))
                    photo = ImageTk.PhotoImage(img)
                    self._grid_refs.append(photo)
                    tk.Label(card, image=photo, bg=card_bg).pack(pady=(5, 2))
                except Exception:
                    tk.Label(card, text='◉', font=('Arial', 40),
                             bg=card_bg, fg=C['sub']).pack(pady=(10, 2))
            else:
                tk.Label(card, text='◉', font=('Arial', 40),
                         bg=card_bg, fg=C['sub']).pack(pady=(10, 2))

            name_col = C['amber'] if sel else (C['green'] if is_live else C['text'])
            tk.Label(card, text=name[:16], font=('Arial', 8, 'bold'),
                     bg=card_bg, fg=name_col, wraplength=140).pack()
            if dept:
                dept_col = DEPT_COLORS.get(dept, C['text2'])
                tk.Label(card, text=dept, font=('Arial', 7),
                         bg=card_bg, fg=dept_col).pack()
            tk.Label(card, text=f'#{pid}', font=('Consolas', 7),
                     bg=card_bg, fg=C['sub']).pack()

            if self._merge_mode:
                handler = lambda e, p=pid: self._toggle_merge_select(p)
            else:
                handler = lambda e, p=pid: self._show_employee_dialog(p)
            for w in [outer, card] + list(card.winfo_children()):
                w.bind('<Button-1>', handler)

    def _show_employee_dialog(self, pid: str):
        row = db.get_person(pid)
        dlg = tk.Toplevel(self.root)
        dlg.title(f'Employee  #{pid}')
        dlg.configure(bg=C['bg'])
        dlg.geometry('520x560')
        dlg.resizable(False, False)
        dlg.grab_set()

        # Photo header
        top = tk.Frame(dlg, bg=C['dim'])
        top.pack(fill='x')
        photo_path = os.path.join(CONFIG.people_dir, pid, 'photo.jpg')
        if os.path.exists(photo_path):
            try:
                img  = Image.open(photo_path).resize((110, 110))
                ph   = ImageTk.PhotoImage(img)
                self._photo_refs.append(ph)
                ph_f = tk.Frame(top, bg=C['bdr_a'])
                ph_f.pack(side='left', padx=16, pady=14)
                tk.Label(ph_f, image=ph, bg=C['bdr_a']).pack(padx=2, pady=2)
            except Exception:
                pass
        meta = tk.Frame(top, bg=C['dim'])
        meta.pack(side='left', fill='x', expand=True, pady=14)
        name_now = (row['name'] if row and row['name'] else '') or f'Person {pid}'
        tk.Label(meta, text=name_now, font=('Arial', 16, 'bold'),
                 bg=C['dim'], fg=C['text'], anchor='w').pack(fill='x')
        tk.Label(meta, text=f'Database ID  #{pid}',
                 font=('Consolas', 9), bg=C['dim'], fg=C['accent'], anchor='w').pack(fill='x', pady=2)
        visits = db.get_visits(pid)
        tk.Label(meta, text=f'{len(visits)} visits recorded',
                 font=('Arial', 9), bg=C['dim'], fg=C['text2'], anchor='w').pack(fill='x')

        _sep(dlg, color=C['bdr_a']).pack(fill='x')

        form = tk.Frame(dlg, bg=C['bg'])
        form.pack(fill='both', expand=True, padx=24, pady=16)

        fields = {}
        field_defs = [
            ('Full Name',    'name',        (row['name'] if row else '') or ''),
            ('Department',   'department',  (row['department'] if row else '') or ''),
            ('Employee ID',  'employee_id', (row['employee_id'] if row else '') or ''),
            ('Role',         'role',        (row['role'] if row else '') or 'employee'),
        ]
        for label, key, default in field_defs:
            r = tk.Frame(form, bg=C['bg'])
            r.pack(fill='x', pady=6)
            tk.Label(r, text=label, font=('Arial', 10),
                     bg=C['bg'], fg=C['text2'], width=14, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            e = tk.Entry(r, textvariable=var, bg=C['dim'], fg=C['text'],
                         insertbackground=C['accent'], relief='flat',
                         font=('Arial', 11),
                         highlightthickness=1, highlightbackground=C['border'],
                         highlightcolor=C['accent'])
            e.pack(side='left', fill='x', expand=True, ipady=6, padx=(8, 0))
            fields[key] = var

        _sep(dlg, color=C['border']).pack(fill='x', pady=4)

        btn_row = tk.Frame(dlg, bg=C['bg'])
        btn_row.pack(fill='x', padx=24, pady=12)

        def _save():
            db.update_person_info(
                pid,
                fields['name'].get().strip(),
                fields['department'].get().strip(),
                fields['employee_id'].get().strip(),
                fields['role'].get().strip(),
            )
            self._refresh_caches()
            self._refresh_employees_grid()
            dlg.destroy()

        def _delete():
            if messagebox.askyesno('Delete', f'Delete Person #{pid}? Cannot be undone.',
                                   parent=dlg):
                self._delete_person_silent(pid)
                self._refresh_caches()
                self._refresh_employees_grid()
                dlg.destroy()

        _btn(btn_row, '💾  Save Changes', _save,
             font=('Arial', 10), padx=18, pady=6).pack(side='left', padx=(0, 8))
        _btn(btn_row, '🗑  Delete', _delete,
             bg=C['red_d'], fg=C['text'], font=('Arial', 10), padx=14, pady=6).pack(side='left')
        _btn(btn_row, 'Cancel', dlg.destroy,
             bg=C['dim'], fg=C['text2'], font=('Arial', 10), padx=14, pady=6).pack(side='right')

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4 — Attendance
    # ══════════════════════════════════════════════════════════════════════════

    def _build_attendance_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['attendance'] = frame

        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        self._att_title = tk.Label(hdr, text='  📋  ATTENDANCE',
                                    font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['text'])
        self._att_title.pack(side='left', pady=14)
        _btn(hdr, '↻  Refresh', self._refresh_attendance,
             bg=C['dim'], fg=C['text2'], font=('Arial', 9), padx=12).pack(side='right', padx=12, pady=12)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        # Stats bar
        stats_bar = tk.Frame(frame, bg=C['panel'])
        stats_bar.pack(fill='x', padx=14, pady=8)
        self._att_stats = tk.Label(stats_bar, text='',
                                    font=('Arial', 10), bg=C['panel'], fg=C['text2'])
        self._att_stats.pack(side='left')

        _sep(frame, color=C['border']).pack(fill='x')

        # Tabs: Inside Now | Full Log
        nb = ttk.Notebook(frame)
        nb.pack(fill='both', expand=True, padx=0, pady=0)

        self._att_inside_frame = tk.Frame(nb, bg=C['panel'])
        self._att_log_frame    = tk.Frame(nb, bg=C['panel'])
        nb.add(self._att_inside_frame, text='  ✅  Currently Inside  ')
        nb.add(self._att_log_frame,    text='  📋  Full Log Today    ')

        # Inside now table
        cols_in = ('#', 'Name', 'Department', 'Employee ID', 'Entry Time', 'Duration')
        self._tree_inside = ttk.Treeview(self._att_inside_frame,
                                          columns=cols_in, show='headings',
                                          style='D.Treeview')
        for col, w in zip(cols_in, [40, 190, 130, 110, 110, 90]):
            self._tree_inside.heading(col, text=col)
            self._tree_inside.column(col, width=w, anchor='center' if col == '#' else 'w')
        self._tree_inside.tag_configure('live', foreground=C['green'])
        sb_in = ttk.Scrollbar(self._att_inside_frame, orient='vertical',
                               command=self._tree_inside.yview)
        self._tree_inside.configure(yscrollcommand=sb_in.set)
        sb_in.pack(side='right', fill='y')
        self._tree_inside.pack(fill='both', expand=True)

        # Full log table
        cols_log = ('#', 'Name', 'Department', 'Action', 'Time')
        self._tree_log = ttk.Treeview(self._att_log_frame,
                                       columns=cols_log, show='headings',
                                       style='D.Treeview')
        for col, w in zip(cols_log, [40, 200, 140, 80, 150]):
            self._tree_log.heading(col, text=col)
            self._tree_log.column(col, width=w, anchor='center' if col in ('#', 'Action') else 'w')
        self._tree_log.tag_configure('ENTRY', foreground=C['green'])
        self._tree_log.tag_configure('EXIT',  foreground=C['red'])
        sb_log = ttk.Scrollbar(self._att_log_frame, orient='vertical',
                                command=self._tree_log.yview)
        self._tree_log.configure(yscrollcommand=sb_log.set)
        sb_log.pack(side='right', fill='y')
        self._tree_log.pack(fill='both', expand=True)

    def _refresh_attendance(self):
        if not hasattr(self, '_tree_inside'):
            return
        today = datetime.now().strftime('%Y-%m-%d')
        self._att_title.config(text=f'  📋  ATTENDANCE  —  {today}')

        counts = db.get_today_counts()
        self._att_stats.config(
            text=f'  ✅  Currently Inside: ?    '
                 f'📥  Entries Today: {counts["entries"]}    '
                 f'📤  Exits Today: {counts["exits"]}')

        # Inside now
        for row in self._tree_inside.get_children():
            self._tree_inside.delete(row)
        inside = db.get_currently_inside()
        self._att_stats.config(
            text=f'  ✅  Currently Inside: {len(inside)}    '
                 f'📥  Entries Today: {counts["entries"]}    '
                 f'📤  Exits Today: {counts["exits"]}')
        now = datetime.now()
        for i, row in enumerate(inside, 1):
            name    = row['name'] or f'Person {row["id"]}'
            dept    = row['department'] or '—'
            emp_id  = row['employee_id'] or '—'
            entry_t = row['entry_time'] or ''
            try:
                et  = datetime.strptime(entry_t, '%Y-%m-%d %H:%M:%S')
                dur = int((now - et).total_seconds() // 60)
                dur_str = f'{dur//60}h {dur%60:02d}m' if dur >= 60 else f'{dur}m'
                entry_str = et.strftime('%H:%M')
            except Exception:
                dur_str, entry_str = '—', entry_t[-8:][:5]
            tag = 'live' if row['id'] in self.tracker.active_pids else ''
            self._tree_inside.insert('', 'end',
                values=(i, name, dept, emp_id, entry_str, dur_str), tags=(tag,))

        # Full log
        for row in self._tree_log.get_children():
            self._tree_log.delete(row)
        activity = db.get_recent_activity(200)
        today_acts = [a for a in activity if (a['time'] or '').startswith(today)]
        for i, a in enumerate(today_acts, 1):
            name = a['name'] or f'ID {a["person_id"]}'
            dept = a['department'] or '—'
            t    = (a['time'] or '')[-8:][:5]
            self._tree_log.insert('', 'end',
                values=(i, name, dept, a['action'], t), tags=(a['action'],))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 5 — Recognition
    # ══════════════════════════════════════════════════════════════════════════

    def _build_recognition_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['recognition'] = frame

        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='  🔍  PERSON RECOGNITION  —  Upload any photo',
                 font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['text']).pack(side='left', pady=14)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        zone_outer, zone = _card(frame, glow=C['bdr_a'])
        zone_outer.pack(fill='x', padx=24, pady=18)

        tk.Label(zone, text='🔍  Identify a Person',
                 font=('Arial', 16, 'bold'), bg=C['card'], fg=C['accent']).pack(pady=(20, 6))
        tk.Label(zone,
                 text='Upload any photo — the system finds the closest match in the database.',
                 font=('Arial', 11), bg=C['card'], fg=C['text2']).pack()

        btn_row = tk.Frame(zone, bg=C['card'])
        btn_row.pack(pady=18)
        _btn(btn_row, '📁   Choose Photo', self._do_recognition,
             font=('Arial', 12), padx=26, pady=8).pack(side='left', padx=8)
        _btn(btn_row, '✕  Clear', self._clear_recognition,
             bg=C['dim'], fg=C['text2'], font=('Arial', 11), padx=18, pady=8).pack(side='left', padx=8)

        self._recog_result_frame = tk.Frame(frame, bg=C['bg'])
        self._recog_result_frame.pack(fill='both', expand=True, padx=24, pady=4)

    def _clear_recognition(self):
        for w in self._recog_result_frame.winfo_children():
            w.destroy()
        self._photo_refs = []

    def _do_recognition(self):
        path = filedialog.askopenfilename(
            title='Select Photo',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp *.webp'), ('All', '*.*')]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Error', 'Cannot load image.', parent=self.root)
            return

        self._clear_recognition()

        # Use lenient detection for uploaded photos (bypass strict filters)
        processed = engine._preprocess(img)
        faces = engine._app.get(processed)

        if not faces:
            tk.Label(self._recog_result_frame,
                     text='⚠  No face detected in the uploaded image.',
                     font=('Arial', 13), bg=C['bg'], fg=C['red']).pack(pady=50)
            return

        face = max(faces, key=lambda f: float(getattr(f, 'det_score', 0)))
        emb  = face.embedding

        pid = score = None
        if engine._embeddings:
            pid_best: dict = {}
            for e, p in zip(engine._embeddings, engine._ids):
                s = engine.cosine_sim(emb, e)
                if s > pid_best.get(p, -1.0):
                    pid_best[p] = s
            pid   = max(pid_best, key=pid_best.get)
            score = pid_best[pid]

        res_outer, result = _card(self._recog_result_frame, glow=C['bdr_a'])
        res_outer.pack(fill='x', pady=8)

        # Uploaded photo
        try:
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb).resize((140, 140))
            ph   = ImageTk.PhotoImage(pil)
            self._photo_refs.append(ph)
            ph_f = tk.Frame(result, bg=C['bdr_a'])
            ph_f.pack(side='left', padx=18, pady=16)
            tk.Label(ph_f, image=ph, bg=C['bdr_a']).pack(padx=2, pady=2)
            tk.Label(ph_f, text='Query', font=('Arial', 7),
                     bg=C['bdr_a'], fg=C['bg']).pack()
        except Exception:
            pass

        info = tk.Frame(result, bg=C['card'])
        info.pack(side='left', fill='x', expand=True, pady=16)

        if pid is None:
            tk.Label(info, text='No people registered yet.',
                     font=('Arial', 13), bg=C['card'], fg=C['sub']).pack(anchor='w')
            return

        person_row = db.get_person(pid)
        name = self._name_cache.get(pid, f'Person {pid}')
        dept = (person_row['department'] if person_row else '') or ''
        emp_id = (person_row['employee_id'] if person_row else '') or ''

        if score >= CONFIG.similarity_threshold:
            icon, col = '✓', C['green']
            conf_text = f'High Confidence  ·  {score:.1%}'
        elif score >= 0.35:
            icon, col = '?', C['yellow']
            conf_text = f'Possible Match  ·  {score:.1%}  (low confidence)'
        else:
            icon, col = '~', C['sub']
            conf_text = f'Closest Match  ·  {score:.1%}  (very low — likely no match)'

        tk.Label(info, text=f'{icon}   {name}',
                 font=('Arial', 18, 'bold'), bg=C['card'], fg=col).pack(anchor='w')
        tk.Label(info, text=conf_text,
                 font=('Arial', 11), bg=C['card'], fg=C['text2']).pack(anchor='w', pady=2)
        if dept:
            tk.Label(info, text=f'🏢  {dept}',
                     font=('Arial', 10), bg=C['card'], fg=C['text2']).pack(anchor='w')
        if emp_id:
            tk.Label(info, text=f'🪪  Employee ID: {emp_id}',
                     font=('Arial', 10), bg=C['card'], fg=C['text2']).pack(anchor='w')
        tk.Label(info, text=f'Database ID: #{pid}',
                 font=('Consolas', 9), bg=C['card'], fg=C['sub']).pack(anchor='w', pady=(4, 0))

        # Registered photo
        db_photo = os.path.join(CONFIG.people_dir, pid, 'photo.jpg')
        if os.path.exists(db_photo):
            try:
                dp  = Image.open(db_photo).resize((90, 90))
                dph = ImageTk.PhotoImage(dp)
                self._photo_refs.append(dph)
                row2 = tk.Frame(info, bg=C['card'])
                row2.pack(anchor='w', pady=(8, 4))
                tk.Label(row2, text='Registered:',
                         font=('Arial', 8), bg=C['card'], fg=C['sub']).pack(anchor='w')
                pw = tk.Frame(row2, bg=col)
                pw.pack(anchor='w', pady=2)
                tk.Label(pw, image=dph, bg=col).pack(padx=2, pady=2)
            except Exception:
                pass

        visits = db.get_visits(pid)
        if visits:
            _sep(info, color=C['border']).pack(fill='x', pady=(10, 6))
            tk.Label(info, text=f'Entry / Exit History  ({len(visits)} records):',
                     font=('Arial', 10, 'bold'), bg=C['card'], fg=C['text']).pack(anchor='w', pady=(0, 4))
            for v in visits[:10]:
                vc   = C['green'] if v['action'] == 'ENTRY' else C['red']
                rf   = tk.Frame(info, bg=C['card'])
                rf.pack(anchor='w', fill='x')
                tk.Frame(rf, bg=vc, width=3).pack(side='left', fill='y')
                tk.Label(rf, text=f'  {v["action"]}   {v["time"]}',
                         font=('Consolas', 9), bg=C['card'], fg=vc).pack(side='left')

        _btn(info, '→  View Full Profile',
             lambda p=pid: (self._switch_page('employees'),
                            self.root.after(100, lambda: self._show_employee_dialog(p))),
             bg=C['accent_d'], fg=C['text'], font=('Arial', 10), padx=14).pack(anchor='w', pady=10)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 6 — Watchlist
    # ══════════════════════════════════════════════════════════════════════════

    def _build_watchlist_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['watchlist'] = frame

        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='  🔔  WATCHLIST  —  Alert when detected',
                 font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['yellow']).pack(side='left', pady=14)
        _btn(hdr, '+  Add Person', self._add_to_watchlist,
             bg=C['amber'], fg=C['bg'], font=('Arial', 10), padx=14).pack(side='right', padx=14, pady=12)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        self._wl_alarm_bar = tk.Frame(frame, bg=C['red_d'], height=0)
        self._wl_alarm_bar.pack(fill='x')
        self._wl_alarm_bar.pack_propagate(False)
        self._wl_alarm_lbl = tk.Label(self._wl_alarm_bar, text='',
                                       font=('Arial', 12, 'bold'), bg=C['red_d'], fg='white')
        self._wl_alarm_lbl.pack(pady=8)

        _, self._wl_inner = _scrollable(frame, bg=C['bg'])

    def _refresh_watchlist_ui(self):
        for w in self._wl_inner.winfo_children():
            w.destroy()
        self._wl_refs = []
        entries = []
        if os.path.exists(CONFIG.watchlist_dir):
            entries = sorted(
                [f for f in os.listdir(CONFIG.watchlist_dir)
                 if os.path.isdir(os.path.join(CONFIG.watchlist_dir, f)) and f.isdigit()],
                key=int)
        if not entries:
            ph = tk.Frame(self._wl_inner, bg=C['bg'])
            ph.pack(pady=60)
            tk.Label(ph, text='🔔', font=('Arial', 42), bg=C['bg'], fg=C['dim']).pack()
            tk.Label(ph, text='No watchlist entries',
                     font=('Arial', 14, 'bold'), bg=C['bg'], fg=C['sub']).pack(pady=6)
            tk.Label(ph, text='Click  "+ Add Person"  to start monitoring someone.',
                     font=('Arial', 11), bg=C['bg'], fg=C['sub']).pack()
            return
        for wid in entries:
            self._add_watchlist_card(wid)

    def _add_watchlist_card(self, wid: str):
        wdir       = os.path.join(CONFIG.watchlist_dir, wid)
        label_path = os.path.join(wdir, 'label.txt')
        label      = (open(label_path, encoding='utf-8').read().strip()
                      if os.path.exists(label_path) else f'Alert #{wid}')
        outer, card = _card(self._wl_inner, glow=C['amber'])
        outer.pack(fill='x', padx=10, pady=6)
        photo_path = os.path.join(wdir, 'photo.jpg')
        if os.path.exists(photo_path):
            try:
                img   = Image.open(photo_path).resize((84, 84))
                photo = ImageTk.PhotoImage(img)
                self._wl_refs.append(photo)
                ph_f = tk.Frame(card, bg=C['amber'])
                ph_f.pack(side='left', padx=14, pady=12)
                tk.Label(ph_f, image=photo, bg=C['amber']).pack(padx=2, pady=2)
            except Exception:
                tk.Label(card, text='⚠', font=('Arial', 34),
                         bg=C['card'], fg=C['yellow']).pack(side='left', padx=14)
        else:
            tk.Label(card, text='⚠', font=('Arial', 34),
                     bg=C['card'], fg=C['yellow']).pack(side='left', padx=14)
        info = tk.Frame(card, bg=C['card'])
        info.pack(side='left', fill='x', expand=True, pady=12)
        tk.Label(info, text=label, font=('Arial', 15, 'bold'),
                 bg=C['card'], fg=C['yellow']).pack(anchor='w')
        tk.Label(info, text=f'Watchlist entry  ·  ID #{wid}',
                 font=('Consolas', 9), bg=C['card'], fg=C['sub']).pack(anchor='w', pady=2)
        tk.Label(info, text=f'Alert threshold: {CONFIG.watchlist_alarm_threshold:.0%} similarity',
                 font=('Arial', 9), bg=C['card'], fg=C['sub']).pack(anchor='w')
        _btn(card, '✕  Remove', lambda w=wid: self._remove_from_watchlist(w),
             bg=C['red_d'], fg=C['text'], font=('Arial', 9), padx=12).pack(
             side='right', padx=14, pady=20)

    def _add_to_watchlist(self):
        path = filedialog.askopenfilename(
            title='Select photo for watchlist',
            filetypes=[('Images', '*.jpg *.jpeg *.png *.bmp'), ('All', '*.*')])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror('Error', 'Cannot load image.', parent=self.root)
            return
        faces = engine._app.get(engine._preprocess(img))
        if not faces:
            messagebox.showerror('No Face', 'No face found in the selected image.', parent=self.root)
            return
        label = simpledialog.askstring('Watchlist Label', 'Name or description:', parent=self.root)
        if not label:
            return
        os.makedirs(CONFIG.watchlist_dir, exist_ok=True)
        existing = [int(f) for f in os.listdir(CONFIG.watchlist_dir)
                    if os.path.isdir(os.path.join(CONFIG.watchlist_dir, f)) and f.isdigit()]
        wid  = str(max(existing) + 1) if existing else '1'
        wdir = os.path.join(CONFIG.watchlist_dir, wid)
        os.makedirs(wdir, exist_ok=True)
        face = max(faces, key=lambda f: float(getattr(f, 'det_score', 0)))
        cv2.imwrite(os.path.join(wdir, 'photo.jpg'), img)
        np.save(os.path.join(wdir, 'embedding.npy'), face.embedding)
        with open(os.path.join(wdir, 'label.txt'), 'w', encoding='utf-8') as f:
            f.write(label.strip())
        self._watchlist_embs.append((face.embedding, label.strip(), wid))
        self._refresh_watchlist_ui()

    def _remove_from_watchlist(self, wid: str):
        wdir = os.path.join(CONFIG.watchlist_dir, wid)
        if os.path.exists(wdir):
            shutil.rmtree(wdir)
        self._watchlist_embs = [(e, l, w) for e, l, w in self._watchlist_embs if w != wid]
        self._refresh_watchlist_ui()

    def _load_watchlist_embs(self):
        self._watchlist_embs = []
        if not os.path.exists(CONFIG.watchlist_dir):
            return
        for wid in sorted(
            [f for f in os.listdir(CONFIG.watchlist_dir)
             if os.path.isdir(os.path.join(CONFIG.watchlist_dir, f)) and f.isdigit()], key=int):
            wdir     = os.path.join(CONFIG.watchlist_dir, wid)
            emb_path = os.path.join(wdir, 'embedding.npy')
            lbl_path = os.path.join(wdir, 'label.txt')
            if not os.path.exists(emb_path):
                continue
            emb   = np.load(emb_path)
            label = (open(lbl_path, encoding='utf-8').read().strip()
                     if os.path.exists(lbl_path) else f'Alert #{wid}')
            self._watchlist_embs.append((emb, label, wid))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 7 — Settings
    # ══════════════════════════════════════════════════════════════════════════

    def _build_settings_page(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])
        self._page_frames['settings'] = frame

        hdr = tk.Frame(frame, bg=C['panel'], height=52)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='  ⚙  SYSTEM SETTINGS',
                 font=('Arial', 13, 'bold'), bg=C['panel'], fg=C['text']).pack(side='left', pady=14)

        _sep(frame, color=C['bdr_a']).pack(fill='x')

        _, scroll = _scrollable(frame, bg=C['bg'])

        sections = [
            ('🎯  Recognition', [
                ('Similarity Threshold',  'similarity_threshold',  0.30, 0.80, '%.2f'),
                ('ID Margin (anti-spoof)', 'id_margin',             0.00, 0.15, '%.2f'),
                ('Min Detection Score',   'min_det_score',         0.40, 0.95, '%.2f'),
                ('Min Frontal Score',     'min_frontal_score',     0.20, 0.90, '%.2f'),
                ('Angle Diversity',       'angle_diversity_threshold', 0.60, 0.95, '%.2f'),
                ('Watchlist Threshold',   'watchlist_alarm_threshold', 0.30, 0.80, '%.2f'),
            ]),
            ('📷  Camera & Detection', [
                ('Detection Interval',  'detection_interval', 1,  15, '%d'),
                ('Blur Threshold',      'blur_threshold',     10, 300, '%.0f'),
                ('Min Face Size (px)',  'min_face_size',      20, 200, '%d'),
                ('Door Position (%)',   'door_position_pct',  0.0, 1.0, '%.2f'),
            ]),
            ('🗂  Registration', [
                ('Confirm Frames',   'confirm_frames',   5,  60, '%d'),
                ('Max Face Angles',  'max_face_angles',  4,  20, '%d'),
                ('Lost Timeout (s)', 'lost_timeout_sec', 1,  30, '%d'),
            ]),
        ]
        self._setting_sliders: dict = {}

        for sect_title, fields in sections:
            sect_outer, sect = _card(scroll, glow=C['border'])
            sect_outer.pack(fill='x', padx=20, pady=(14, 4))

            sh = tk.Frame(sect, bg=C['card'])
            sh.pack(fill='x', padx=16, pady=(12, 8))
            tk.Label(sh, text=sect_title, font=('Arial', 11, 'bold'),
                     bg=C['card'], fg=C['accent']).pack(side='left')
            _sep(sect, color=C['border']).pack(fill='x', padx=8)

            for label, key, lo, hi, fmt in fields:
                row = tk.Frame(sect, bg=C['card'])
                row.pack(fill='x', padx=16, pady=5)

                tk.Label(row, text=label, font=('Arial', 10),
                         bg=C['card'], fg=C['text'], width=26, anchor='w').pack(side='left')

                cur  = getattr(CONFIG, key)
                var  = tk.StringVar(value=fmt % cur)
                vlbl = tk.Label(row, textvariable=var,
                                font=('Consolas', 10, 'bold'),
                                bg=C['card'], fg=C['yellow'], width=7)
                vlbl.pack(side='right', padx=(8, 16))

                def _on_slide(v, k=key, f=fmt, sv=var):
                    val = float(v)
                    t   = type(getattr(CONFIG, k))
                    setattr(CONFIG, k, t(val))
                    sv.set(f % val)

                sl = ttk.Scale(row, from_=lo, to=hi, orient='horizontal',
                               command=_on_slide, style='Horizontal.TScale')
                sl.set(cur)
                sl.pack(side='left', fill='x', expand=True)
                self._setting_sliders[key] = sl

            sect_outer.pack_configure(pady=(14, 4))

        # Buttons
        btn_row = tk.Frame(scroll, bg=C['bg'])
        btn_row.pack(fill='x', padx=20, pady=16)

        _btn(btn_row, '💾  Save to File', self._save_settings,
             font=('Arial', 11), padx=22, pady=8).pack(side='left', padx=(0, 10))
        _btn(btn_row, '↺  Reset Defaults', self._reset_settings,
             bg=C['dim'], fg=C['text2'], font=('Arial', 11), padx=18, pady=8).pack(side='left')

        tk.Label(scroll,
                 text='⚠  Changes to "Detection Size" and "Model" require restarting the app.',
                 font=('Arial', 9), bg=C['bg'], fg=C['sub']).pack(padx=20, pady=(0, 20))

    def _save_settings(self):
        data = {
            'similarity_threshold':       CONFIG.similarity_threshold,
            'id_margin':                  CONFIG.id_margin,
            'min_det_score':              CONFIG.min_det_score,
            'min_frontal_score':          CONFIG.min_frontal_score,
            'angle_diversity_threshold':  CONFIG.angle_diversity_threshold,
            'watchlist_alarm_threshold':  CONFIG.watchlist_alarm_threshold,
            'detection_interval':         CONFIG.detection_interval,
            'blur_threshold':             CONFIG.blur_threshold,
            'min_face_size':              CONFIG.min_face_size,
            'door_position_pct':          CONFIG.door_position_pct,
            'confirm_frames':             CONFIG.confirm_frames,
            'max_face_angles':            CONFIG.max_face_angles,
            'lost_timeout_sec':           CONFIG.lost_timeout_sec,
        }
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'settings_override.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo('Saved', f'Settings saved to:\n{save_path}', parent=self.root)

    def _reset_settings(self):
        if not messagebox.askyesno('Reset', 'Reset all settings to default values?',
                                   parent=self.root):
            return
        defaults = {
            'similarity_threshold': 0.52, 'id_margin': 0.04,
            'min_det_score': 0.70, 'min_frontal_score': 0.50,
            'angle_diversity_threshold': 0.80, 'watchlist_alarm_threshold': 0.52,
            'detection_interval': 5, 'blur_threshold': 60.0,
            'min_face_size': 50, 'door_position_pct': 0.5,
            'confirm_frames': 30, 'max_face_angles': 12, 'lost_timeout_sec': 5,
        }
        for key, val in defaults.items():
            setattr(CONFIG, key, val)
            if key in self._setting_sliders:
                self._setting_sliders[key].set(val)

    # ══════════════════════════════════════════════════════════════════════════
    # Employee merge helpers (for Employees page)
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_merge_mode(self):
        self._merge_mode     = not self._merge_mode
        self._merge_selected = []
        if self._merge_mode:
            self._btn_emp_merge.config(text='✕  Cancel Merge', bg=C['red_d'], fg=C['text'])
            self._lbl_emp_status.config(text='Select 2 employees to merge', fg=C['yellow'])
        else:
            self._btn_emp_merge.config(text='⛙  Merge Mode', bg=C['dim'], fg=C['text2'])
            self._btn_do_merge.pack_forget()
            self._lbl_emp_status.config(text='', fg=C['green'])
        self._refresh_employees_grid()

    def _toggle_merge_select(self, pid: str):
        if pid in self._merge_selected:
            self._merge_selected.remove(pid)
        elif len(self._merge_selected) < 2:
            self._merge_selected.append(pid)
        if len(self._merge_selected) == 2:
            self._btn_do_merge.pack(side='right', padx=4)
            self._lbl_emp_status.config(
                text=f'#{self._merge_selected[0]}  +  #{self._merge_selected[1]}', fg=C['amber'])
        else:
            self._btn_do_merge.pack_forget()
            sel = self._merge_selected[0] if self._merge_selected else '—'
            self._lbl_emp_status.config(text=f'Selected: #{sel}  — pick one more', fg=C['yellow'])
        self._refresh_employees_grid()

    def _do_manual_merge(self):
        if len(self._merge_selected) != 2:
            return
        pid_a, pid_b = self._merge_selected
        va = len(db.get_visits(pid_a))
        vb = len(db.get_visits(pid_b))
        keep   = pid_a if va >= vb else pid_b
        victim = pid_b if va >= vb else pid_a
        nk = self._name_cache.get(keep,   f'Person {keep}')
        nv = self._name_cache.get(victim, f'Person {victim}')
        if not messagebox.askyesno('Confirm Merge',
                                   f'Merge  "{nv}"  →  "{nk}" ?\n\nAll records transfer to #{keep}.'
                                   f'\n#{victim} will be deleted.', parent=self.root):
            return
        for emb in [e for e, i in zip(engine._embeddings, engine._ids) if i == victim]:
            engine.add_angle(keep, emb)
        with db._lock:
            c = db._conn()
            c.execute('UPDATE visits SET person_id=? WHERE person_id=?', (keep, victim))
            c.execute('UPDATE seen   SET person_id=? WHERE person_id=?', (keep, victim))
            c.commit()
        self._delete_person_silent(victim)
        self._merge_mode     = False
        self._merge_selected = []
        self._btn_emp_merge.config(text='⛙  Merge Mode', bg=C['dim'], fg=C['text2'])
        self._btn_do_merge.pack_forget()
        self._lbl_emp_status.config(text=f'✓  Merged into #{keep}', fg=C['green'])
        self._refresh_caches()
        self._refresh_employees_grid()

    # ══════════════════════════════════════════════════════════════════════════
    # Alarm
    # ══════════════════════════════════════════════════════════════════════════

    def _trigger_alarm(self, label: str, sim: float):
        now = datetime.now()
        if not self._last_alarm_time or \
                (now - self._last_alarm_time).total_seconds() >= 15:
            self._last_alarm_time = now
            self.root.bell()
        self._alarm_active = True
        self._alarm_lbl.config(text=f'▲  ALERT: {label}  ({sim:.0%})')
        if self._current_page == 'watchlist':
            self._wl_alarm_bar.config(height=46)
            self._wl_alarm_lbl.config(
                text=f'▲   WATCHLIST MATCH:  {label}   —  {sim:.0%} confidence')

    def _clear_alarm_ui(self):
        self._alarm_lbl.config(text='')
        self._wl_alarm_bar.config(height=0)
        self._wl_alarm_lbl.config(text='')

    # ══════════════════════════════════════════════════════════════════════════
    # Camera
    # ══════════════════════════════════════════════════════════════════════════

    def _start_camera(self):
        w, h = CONFIG.resolutions[self._res_idx]
        self._cap = cv2.VideoCapture(CONFIG.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cam_running = True
        self._lbl_res.config(text=f'{w}×{h}')
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        frame_idx    = 0
        cached_faces = []
        while self._cam_running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            frame_idx += 1
            if frame_idx % CONFIG.detection_interval == 0:
                cached_faces = engine.detect(frame)
                self._check_watchlist_matches(cached_faces)

            h, w   = frame.shape[:2]
            door_x = int(w * CONFIG.door_position_pct)
            draws  = self.tracker.update(cached_faces, frame, door_x)

            self._fps_cnt += 1
            elapsed = (datetime.now() - self._fps_ts).total_seconds()
            if elapsed >= 1.0:
                self._fps     = self._fps_cnt / elapsed
                self._fps_cnt = 0
                self._fps_ts  = datetime.now()

            annotated = self._annotate(frame, draws, w, h, door_x)
            if not self._frame_q.full():
                self._frame_q.put(annotated)

    def _check_watchlist_matches(self, faces: list):
        if not self._watchlist_embs:
            return
        found = False
        best_label, best_sim = '', 0.0
        for face in faces:
            emb = face.embedding
            for w_emb, w_label, wid in self._watchlist_embs:
                sim = engine.cosine_sim(emb, w_emb)
                if sim >= CONFIG.watchlist_alarm_threshold and sim > best_sim:
                    best_sim, best_label, found = sim, w_label, True
        if found:
            self._alarm_info      = {'label': best_label, 'sim': best_sim}
            self._alarm_last_seen = datetime.now()

    def _annotate(self, frame, draws, w, h, door_x):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (4, 8, 18), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        if self._alarm_active:
            cv2.rectangle(frame, (0, 0), (w, 52), (120, 0, 18), -1)
            cv2.putText(frame, '  ▲  ALERT  —  WATCHLIST PERSON DETECTED',
                        (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

        cv2.line(frame,
                 (door_x, 52 if self._alarm_active else 0),
                 (door_x, h - 40), (0, 196, 240), 1)

        cv2.putText(frame, '<- ENTRY',
                    (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 1)
        cv2.putText(frame, 'EXIT ->',
                    (door_x + 8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 61, 90), 1)

        ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        cv2.putText(frame, ts,
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 140, 200), 1)
        fps_txt = f'FPS {self._fps:.0f}'
        (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, fps_txt,
                    (w - fw - 10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 214, 10), 1)

        for bbox, pid, confirmed, rem, score in draws:
            x1, y1, x2, y2 = bbox
            if confirmed:
                color = (0, 230, 118)
                name  = self._name_cache.get(pid, f'Person {pid}')
                dept  = self._dept_cache.get(pid, '')
                label = f'{name}  {score:.0%}' if score > 0 else name
                if dept:
                    label = f'{label}  [{dept}]'
            else:
                color = (0, 196, 240)
                label = f'Scanning  {rem}'

            L = 20
            for pts in [((x1, y1), (x1+L, y1)), ((x1, y1), (x1, y1+L)),
                        ((x2-L, y1), (x2, y1)), ((x2, y1), (x2, y1+L)),
                        ((x1, y2-L), (x1, y2)), ((x1, y2), (x1+L, y2)),
                        ((x2-L, y2), (x2, y2)), ((x2, y2-L), (x2, y2))]:
                cv2.line(frame, pts[0], pts[1], color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1 - 4, th + 14)
            cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 14, ty - 2), color, -1)
            cv2.putText(frame, label, (x1 + 7, ty - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (4, 8, 18), 2)

        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # Display & refresh loops
    # ══════════════════════════════════════════════════════════════════════════

    def _update_display(self):
        try:
            frame = self._frame_q.get_nowait()
            cw = self._cam_canvas.winfo_width()
            ch = self._cam_canvas.winfo_height()
            if cw > 1 and ch > 1:
                fh, fw = frame.shape[:2]
                scale  = min(cw / fw, ch / fh)
                nw, nh = int(fw * scale), int(fh * scale)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img    = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
                photo  = ImageTk.PhotoImage(img)
                ox, oy = (cw - nw) // 2, (ch - nh) // 2
                self._cam_canvas.delete('all')
                self._cam_canvas.create_image(ox, oy, anchor='nw', image=photo)
                self._cam_canvas._photo = photo
            self._lbl_fps.config(text=f'{self._fps:.0f}')
            self._lbl_active.config(text=str(len(self.tracker.active_pids)))
        except queue.Empty:
            pass

        alarm = self._alarm_info
        if alarm:
            self._alarm_info = None
            self._trigger_alarm(alarm['label'], alarm['sim'])
        if self._alarm_active and self._alarm_last_seen is not None:
            if (datetime.now() - self._alarm_last_seen).total_seconds() > 3.0:
                self._alarm_active    = False
                self._alarm_last_seen = None
                self._clear_alarm_ui()

        self.root.after(CONFIG.camera_refresh_ms, self._update_display)

    def _periodic_refresh(self):
        active = set(self.tracker.active_pids)
        if active != self._last_active_pids:
            self._last_active_pids = active
            self._kpi_live.set(str(len(active)))
            if self._current_page == 'live':
                self._refresh_live_list()

        if self._current_page == 'dashboard':
            self._refresh_dashboard()
        elif self._current_page == 'attendance':
            self._refresh_attendance()

        self.root.after(CONFIG.stats_refresh_ms, self._periodic_refresh)

    def _schedule_loops(self):
        self._update_display()
        self._periodic_refresh()

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_caches(self):
        rows = db.get_all_people()
        self._name_cache = {r['id']: (r['name'] or f'Person {r["id"]}') for r in rows}
        self._dept_cache = {r['id']: (r['department'] or '') for r in rows}

    def _delete_person_silent(self, pid: str):
        d = os.path.join(CONFIG.people_dir, pid)
        if os.path.exists(d):
            shutil.rmtree(d)
        db.delete_person(pid)
        engine.remove_person(pid)
        self._name_cache.pop(pid, None)
        self._dept_cache.pop(pid, None)

    def _set_res(self, idx):
        self._res_idx = idx
        w, h = CONFIG.resolutions[idx]
        if self._cap:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._lbl_res.config(text=f'{w}×{h}')

    def _res_up(self):
        if self._res_idx < len(CONFIG.resolutions) - 1:
            self._set_res(self._res_idx + 1)

    def _res_down(self):
        if self._res_idx > 0:
            self._set_res(self._res_idx - 1)

    def run(self):
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._cam_running = False
        if self._cap:
            self._cap.release()
        self.root.destroy()
