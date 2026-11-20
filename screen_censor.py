            duplicate = False
            for old in kept:
                ox1,oy1,ox2,oy2,_ = old
                ocx,ocy = (ox1+ox2)/2,(oy1+oy2)/2
                if abs(cx-ocx) < min(x2-x1, ox2-ox1)*0.45 and abs(cy-ocy) < min(y2-y1, oy2-oy1)*0.45:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept

    def _scan_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary display
            while self.running and not self.stop_event.is_set():
                started = time.perf_counter()
                raw = np.asarray(sct.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces on a downscaled frame for speed, then restore coordinates.
                scale = 0.5 if max(w, h) >= 1400 else 0.75
                small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                faces = self.face_cascade.detectMultiScale(
                    small, scaleFactor=1.08, minNeighbors=4, minSize=(22, 22)
                )
                boxes = []
                inv = 1.0 / scale
                for x,y,fw,fh in faces:
                    x1,y1 = int(x*inv), int(y*inv)
                    x2,y2 = int((x+fw)*inv), int((y+fh)*inv)
                    boxes.append((x1,y1,x2,y2,"losser"))
                boxes.extend(self._nudenet_boxes(frame))
                # Prepare blurred screen patches. The live overlay renderer only
                # redraws these cached patches at 30 FPS; AI inference runs slower.
                patches = []
                screen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for x1,y1,x2,y2,label in boxes:
                    x1,y1 = max(0,x1),max(0,y1)
                    x2,y2 = min(w,x2),min(h,y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    patch = screen.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
                    patches.append((x1,y1,patch,label))
                with self.lock:
                    self.boxes = boxes
                    self.patch_data = patches
                    self.frame_size = (w,h)
                elapsed = time.perf_counter() - started
                time.sleep(max(0.02, DETECTION_INTERVAL - elapsed))

    def refresh_overlay(self):
        if self.running and self.overlay.state() != "withdrawn":
            with self.lock:
                patches = list(self.patch_data)
            self.canvas.delete("all")
            self.patch_images = []
            for x,y,patch,label in patches:
                photo = ImageTk.PhotoImage(patch)
                self.patch_images.append(photo)
                self.canvas.create_image(x, y, image=photo, anchor="nw")
                self.canvas.create_rectangle(x, y, x+patch.width, y+patch.height,
                                             outline="#ff4d6d", width=2)
                tag_h = 22
                self.canvas.create_rectangle(x, y, x+70, y+tag_h,
                                             fill="#151923", outline="#ff4d6d", width=1)
                self.canvas.create_text(x+35, y+tag_h/2, text="losser",
                                        fill="white", font=("Segoe UI", 10, "bold"))
            self.overlay.lift()
        self.root.after(OVERLAY_REFRESH_MS, self.refresh_overlay)

    def _failed(self, error):
        self.status.config(text="Sensor gagal / berhenti", fg="#ff7777")
        messagebox.showerror(APP_TITLE, "Gagal menjalankan sensor:\n\n" + error)

    def _stopped_ui(self):
        self.running = False
        try:
            self.overlay.withdraw()
        except Exception:
            pass
        with self.lock:
            self.boxes = []
            self.patch_data = []
        self.canvas.delete("all")
        self.patch_images = []
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if not self.closing:
            self.status.config(text="Status: Mati", fg="#ff7777")

    def stop(self):
        self.running = False
        self.stop_event.set()
        self._stopped_ui()

    def close(self):
        self.closing = True
        self.stop()
        if self.hotkey_registered:
            try:
                ctypes.windll.user32.UnregisterHotKey(None, 0x5343)
            except Exception:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenCensor(root)
    root.mainloop()
