import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sqlite3
import os
from PIL import Image, ImageTk
from datetime import datetime

PEOPLE_FOLDER = 'people'
DB_PATH = 'data.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

class Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title('Security Dashboard')
        self.root.geometry('1200x700')
        self.root.configure(bg='#1e1e1e')
        self.selected_person = None
        self.photo_refs = []
        self.build_ui()
        self.load_people()
        self.auto_refresh()

    def auto_refresh(self):
        self.load_people()
        self.root.after(10000, self.auto_refresh)

    def build_ui(self):
        header = tk.Frame(self.root, bg='#2d2d2d', pady=10)
        header.pack(fill='x')

        tk.Label(header, text='Security Dashboard', font=('Arial', 20, 'bold'),
                 bg='#2d2d2d', fg='white').pack(side='left', padx=20)

        self.stats_label = tk.Label(header, text='', font=('Arial', 12),
                                     bg='#2d2d2d', fg='#aaaaaa')
        self.stats_label.pack(side='right', padx=20)

        search_frame = tk.Frame(self.root, bg='#1e1e1e', pady=10)
        search_frame.pack(fill='x', padx=20)

        tk.Label(search_frame, text='Search:', bg='#1e1e1e', fg='white',
                 font=('Arial', 12)).pack(side='left')

        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.on_search)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                font=('Arial', 12), bg='#3d3d3d', fg='white',
                                insertbackground='white', width=30)
        search_entry.pack(side='left', padx=10)

        tk.Button(search_frame, text='Refresh', command=self.load_people,
                  bg='#0078d4', fg='white', font=('Arial', 11),
                  relief='flat', padx=15).pack(side='right')

        body = tk.Frame(self.root, bg='#1e1e1e')
        body.pack(fill='both', expand=True, padx=20, pady=10)

        left = tk.Frame(body, bg='#2d2d2d', width=400)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text='People', font=('Arial', 14, 'bold'),
                 bg='#2d2d2d', fg='white', pady=10).pack()

        canvas = tk.Canvas(left, bg='#2d2d2d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient='vertical', command=canvas.yview)
        self.people_frame = tk.Frame(canvas, bg='#2d2d2d')

        self.people_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        canvas.create_window((0, 0), window=self.people_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self.right = tk.Frame(body, bg='#2d2d2d')
        self.right.pack(side='left', fill='both', expand=True)

        tk.Label(self.right, text='Select a person to view details',
                 font=('Arial', 14), bg='#2d2d2d', fg='#aaaaaa').pack(expand=True)

    def load_people(self):
        for widget in self.people_frame.winfo_children():
            widget.destroy()
        self.photo_refs = []

        if not os.path.exists(PEOPLE_FOLDER):
            return

        people = sorted([f for f in os.listdir(PEOPLE_FOLDER)
                        if os.path.isdir(f'{PEOPLE_FOLDER}/{f}') and f.isdigit()],
                       key=lambda x: int(x))

        try:
            cursor.execute('SELECT COUNT(*) FROM visits')
            total_visits = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM visits WHERE date(time) = date('now')")
            today_visits = cursor.fetchone()[0]
            self.stats_label.config(
                text=f'Total People: {len(people)} | Total Visits: {total_visits} | Today: {today_visits}')
        except:
            pass

        search = self.search_var.get().lower()

        for pid in people:
            info_path = f'{PEOPLE_FOLDER}/{pid}/info.txt'
            name = pid
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if line.startswith('Name:'):
                            name = line.replace('Name:', '').strip()

            if search and search not in pid.lower() and search not in name.lower():
                continue

            self.add_person_card(pid, name)

    def add_person_card(self, pid, name):
        card = tk.Frame(self.people_frame, bg='#3d3d3d', cursor='hand2')
        card.pack(fill='x', padx=5, pady=3)

        photo_path = f'{PEOPLE_FOLDER}/{pid}/photo.jpg'
        if os.path.exists(photo_path):
            try:
                img = Image.open(photo_path)
                img = img.resize((50, 50))
                photo = ImageTk.PhotoImage(img)
                self.photo_refs.append(photo)
                tk.Label(card, image=photo, bg='#3d3d3d').pack(side='left', padx=5, pady=5)
            except:
                tk.Label(card, text='👤', font=('Arial', 20),
                         bg='#3d3d3d', fg='white').pack(side='left', padx=5)
        else:
            tk.Label(card, text='👤', font=('Arial', 20),
                     bg='#3d3d3d', fg='white').pack(side='left', padx=5)

        info_frame = tk.Frame(card, bg='#3d3d3d')
        info_frame.pack(side='left', fill='x', expand=True, padx=5)

        tk.Label(info_frame, text=f'Person {pid}' if name == pid else name,
                 font=('Arial', 12, 'bold'), bg='#3d3d3d', fg='white').pack(anchor='w')

        try:
            cursor.execute('SELECT COUNT(*) FROM visits WHERE person_id = ?', (pid,))
            count = cursor.fetchone()[0]
            cursor.execute('SELECT time FROM visits WHERE person_id = ? ORDER BY id DESC LIMIT 1', (pid,))
            last = cursor.fetchone()
            last_time = last[0] if last else 'Never'
            cursor.execute('SELECT time FROM seen WHERE person_id = ? ORDER BY id DESC LIMIT 1', (pid,))
            last_seen = cursor.fetchone()
            last_seen_time = last_seen[0] if last_seen else 'Never'
        except:
            count = 0
            last_time = 'Never'
            last_seen_time = 'Never'

        tk.Label(info_frame, text=f'Visits: {count} | Last visit: {last_time}',
                 font=('Arial', 10), bg='#3d3d3d', fg='#aaaaaa').pack(anchor='w')
        tk.Label(info_frame, text=f'Last seen: {last_seen_time}',
                 font=('Arial', 10), bg='#3d3d3d', fg='#ffaa00').pack(anchor='w')

        card.bind('<Button-1>', lambda e, p=pid: self.show_details(p))
        for widget in card.winfo_children():
            widget.bind('<Button-1>', lambda e, p=pid: self.show_details(p))
            for w in widget.winfo_children():
                w.bind('<Button-1>', lambda e, p=pid: self.show_details(p))

    def show_details(self, pid):
        for widget in self.right.winfo_children():
            widget.destroy()

        self.selected_person = pid

        header = tk.Frame(self.right, bg='#2d2d2d')
        header.pack(fill='x', padx=10, pady=10)

        photo_path = f'{PEOPLE_FOLDER}/{pid}/photo.jpg'
        if os.path.exists(photo_path):
            try:
                img = Image.open(photo_path)
                img = img.resize((100, 100))
                photo = ImageTk.PhotoImage(img)
                self.photo_refs.append(photo)
                tk.Label(header, image=photo, bg='#2d2d2d').pack(side='left', padx=10)
            except:
                pass

        info = tk.Frame(header, bg='#2d2d2d')
        info.pack(side='left', padx=10)

        name = pid
        info_path = f'{PEOPLE_FOLDER}/{pid}/info.txt'
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    if line.startswith('Name:'):
                        name = line.replace('Name:', '').strip()

        tk.Label(info, text=f'Person {pid}' if name == pid else f'{name} (Person {pid})',
                 font=('Arial', 16, 'bold'), bg='#2d2d2d', fg='white').pack(anchor='w')

        try:
            cursor.execute('SELECT COUNT(*) FROM visits WHERE person_id = ?', (pid,))
            count = cursor.fetchone()[0]
            cursor.execute('SELECT time FROM seen WHERE person_id = ? ORDER BY id DESC LIMIT 1', (pid,))
            last_seen = cursor.fetchone()
            last_seen_time = last_seen[0] if last_seen else 'Never'
        except:
            count = 0
            last_seen_time = 'Never'

        tk.Label(info, text=f'Total Visits: {count}',
                 font=('Arial', 12), bg='#2d2d2d', fg='#aaaaaa').pack(anchor='w')
        tk.Label(info, text=f'Last seen: {last_seen_time}',
                 font=('Arial', 12), bg='#2d2d2d', fg='#ffaa00').pack(anchor='w')

        btn_frame = tk.Frame(header, bg='#2d2d2d')
        btn_frame.pack(side='right', padx=10)

        tk.Button(btn_frame, text='Rename', command=lambda: self.rename_person(pid),
                  bg='#0078d4', fg='white', font=('Arial', 11),
                  relief='flat', padx=10, pady=5).pack(pady=3)

        tk.Button(btn_frame, text='Delete', command=lambda: self.delete_person(pid),
                  bg='#c42b1c', fg='white', font=('Arial', 11),
                  relief='flat', padx=10, pady=5).pack(pady=3)

        # تب ها
        notebook = ttk.Notebook(self.right)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # تب ورود و خروج
        visits_frame = tk.Frame(notebook, bg='#2d2d2d')
        notebook.add(visits_frame, text='Visits')

        cols = ('ID', 'Action', 'Time')
        tree1 = ttk.Treeview(visits_frame, columns=cols, show='headings', height=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background='#3d3d3d', foreground='white',
                        fieldbackground='#3d3d3d', font=('Arial', 11))
        style.configure('Treeview.Heading', background='#2d2d2d', foreground='white',
                        font=('Arial', 11, 'bold'))

        for col in cols:
            tree1.heading(col, text=col)
            tree1.column(col, width=150)

        try:
            cursor.execute('SELECT id, action, time FROM visits WHERE person_id = ? ORDER BY id DESC', (pid,))
            for row in cursor.fetchall():
                tree1.insert('', 'end', values=row)
        except:
            pass

        sb1 = ttk.Scrollbar(visits_frame, orient='vertical', command=tree1.yview)
        tree1.configure(yscrollcommand=sb1.set)
        sb1.pack(side='right', fill='y')
        tree1.pack(fill='both', expand=True)

        # تب ساعت دیده شدن
        seen_frame = tk.Frame(notebook, bg='#2d2d2d')
        notebook.add(seen_frame, text='Seen History')

        cols2 = ('ID', 'Time')
        tree2 = ttk.Treeview(seen_frame, columns=cols2, show='headings', height=10)

        for col in cols2:
            tree2.heading(col, text=col)
            tree2.column(col, width=200)

        try:
            cursor.execute('SELECT id, time FROM seen WHERE person_id = ? ORDER BY id DESC', (pid,))
            for row in cursor.fetchall():
                tree2.insert('', 'end', values=row)
        except:
            pass

        sb2 = ttk.Scrollbar(seen_frame, orient='vertical', command=tree2.yview)
        tree2.configure(yscrollcommand=sb2.set)
        sb2.pack(side='right', fill='y')
        tree2.pack(fill='both', expand=True)

    def rename_person(self, pid):
        new_name = simpledialog.askstring('Rename', f'Enter name for Person {pid}:')
        if new_name:
            info_path = f'{PEOPLE_FOLDER}/{pid}/info.txt'
            lines = []
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            lines = [l for l in lines if not l.startswith('Name:')]
            lines.insert(1, f'Name: {new_name}\n')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            self.load_people()
            self.show_details(pid)

    def delete_person(self, pid):
        if messagebox.askyesno('Delete', f'Are you sure you want to delete Person {pid}?'):
            import shutil
            shutil.rmtree(f'{PEOPLE_FOLDER}/{pid}')
            try:
                cursor.execute('DELETE FROM visits WHERE person_id = ?', (pid,))
                cursor.execute('DELETE FROM seen WHERE person_id = ?', (pid,))
                conn.commit()
            except:
                pass
            self.load_people()
            for widget in self.right.winfo_children():
                widget.destroy()
            tk.Label(self.right, text='Select a person to view details',
                     font=('Arial', 14), bg='#2d2d2d', fg='#aaaaaa').pack(expand=True)

    def on_search(self, *args):
        self.load_people()

root = tk.Tk()
app = Dashboard(root)
root.mainloop()
conn.close()