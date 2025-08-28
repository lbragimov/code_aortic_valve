import tkinter as tk
from tkinter import messagebox
import re


def sort_cases(cases):
    def key_func(case):
        match = re.match(r'([a-zA-Z]+)(\d+)', case)
        if match:
            prefix = match.group(1)   # буквы
            number = int(match.group(2))  # число
            return (prefix, number)
        return (case, float("inf"))
    return sorted(cases, key=key_func)


def center_window(root, width=400, height=400):
    # ширина/высота экрана
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # вычисляем координаты для центра
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    root.geometry(f"{width}x{height}+{x}+{y}")

class CaseSelector:
    def __init__(self, cases):
        self.cases = sort_cases(cases)
        self.selected = None

    def run(self):
        root = tk.Tk()
        root.title("Case Selection")

        # центрируем окно
        center_window(root, width=400, height=400)

        # подпись
        label = tk.Label(root, text="Case Selection:", font=("Arial", 12))
        label.pack(pady=10)

        # case list (scrollable listbox)
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame, orient="vertical")
        listbox = tk.Listbox(frame, selectmode=tk.SINGLE, width=40, height=15,
                             yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        # заполняем список
        for case in self.cases:
            listbox.insert(tk.END, case)

        # кнопка подтверждения
        def confirm():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "First select a case!")
                return
            self.selected = listbox.get(selection[0])
            messagebox.showinfo("Selection made", f"You selected: {self.selected}")
            root.destroy()

        button = tk.Button(root, text="Select", command=confirm)
        button.pack(pady=10)

        root.mainloop()
        return self.selected