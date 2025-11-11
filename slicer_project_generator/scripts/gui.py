import tkinter as tk
from tkinter import messagebox
import re
import os


def ask_add_gh_results(folder_path):
    """Окно выбора .nii.gz файла для добавления GH результатов.
       Возвращает путь к выбранному файлу или None, если отмена.
    """
    # привязываем scrollregion к содержимому
    def update_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    # поддержка прокрутки колесом
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # при прокрутке колесом — обновляем положение ползунка
    def on_scroll(*args):
        canvas.yview(*args)
        scrollbar.set(*args)

    # кнопки
    def confirm():
        nonlocal selected_file
        chosen = [f for var, f in vars_ if var.get()]
        if len(chosen) == 0:
            messagebox.showinfo("No selection", "No file selected.")
            selected_file = None
        elif len(chosen) > 1:
            messagebox.showwarning("Multiple selected", "Select only one file.")
            return
        else:
            selected_file = os.path.join(folder_path, chosen[0])
        root.destroy()

    def cancel():
        nonlocal selected_file
        selected_file = None
        root.destroy()

    # собираем список файлов
    files = [f for f in os.listdir(folder_path) if f.endswith(".nii.gz")]
    if not files:
        messagebox.showinfo("No Files", f"No .nii.gz files found in:\n{folder_path}")
        return None

    selected_file = None  # сюда запишем результат

    # создаем окно
    root = tk.Tk()
    root.title("Select GH Result File")

    # центрируем окно
    root.geometry("400x400")
    # root.resizable(False, False)

    # подпись
    tk.Label(root, text="Select a GH result file:", font=("Arial", 11))
    # label.pack(pady=10)

    # скроллируемый список с чекбоксами
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # scrollable_frame.bind(
    #     "<Configure>",
    #     lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    # )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollable_frame.bind("<Configure>", update_scrollregion)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # создаем чекбоксы
    vars_ = []
    list_files = sort_cases(files)
    for f in list_files:
        var = tk.BooleanVar()
        chk = tk.Checkbutton(scrollable_frame, text=f[:-7], variable=var)  # убираем .nii.gz
        chk.pack(anchor="w", pady=2)
        vars_.append((var, f))

    canvas.configure(yscrollcommand=on_scroll)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="OK", command=confirm).pack(side="left", padx=10)
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side="left", padx=10)

    root.mainloop()
    return selected_file


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
    def __init__(self, cases, gh_lines_pred_mask_folder):
        self.cases = sort_cases(cases)
        self.gh_lines_pred_mask_folder = gh_lines_pred_mask_folder
        self.selected = None
        self.add_gh_results = False  # новое поле для хранения ответа

    def run(self):

        # сначала показываем окно с вопросом
        self.add_gh_results = ask_add_gh_results(self.gh_lines_pred_mask_folder)

        if self.add_gh_results:
            filename = os.path.basename(self.add_gh_results)[:-7]
            return filename, self.add_gh_results
        else:
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
            return self.selected, None