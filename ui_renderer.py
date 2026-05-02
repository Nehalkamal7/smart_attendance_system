"""
Advanced UI Renderer - Modern Dark Theme with Professional Overlays
Provides a visually stunning interface for the Smart Attendance System.
"""
import cv2
import numpy as np
import time
from datetime import datetime


# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":        (15, 15, 25),
    "bg_card":        (25, 28, 45),
    "bg_card2":       (30, 35, 55),
    "accent":         (99, 212, 255),       # Cyan-blue
    "accent2":        (130, 80, 255),       # Purple
    "accent3":        (80, 230, 180),       # Teal-green
    "success":        (60, 210, 120),
    "warning":        (255, 180, 50),
    "danger":         (255, 80, 80),
    "white":          (255, 255, 255),
    "gray_light":     (180, 185, 200),
    "gray_mid":       (100, 108, 130),
    "gray_dark":      (50, 55, 75),
    "face_known":     (60, 215, 120),       # Green
    "face_unknown":   (80, 80, 255),        # Red-ish (BGR)
    "face_corner":    (99, 212, 255),       # Cyan corners
    "overlay_bg":     (10, 12, 22),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def alpha_blend(frame, overlay_color, alpha, rect):
    """Blend a colored rectangle onto the frame."""
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    colored = np.full_like(roi, overlay_color)
    blended = cv2.addWeighted(roi, 1 - alpha, colored, alpha, 0)
    frame[y1:y2, x1:x2] = blended


def draw_rounded_rect(frame, x1, y1, x2, y2, color, thickness=2, radius=10, filled=False):
    """Draw a rounded rectangle."""
    if filled:
        # Corners
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
        # Rectangles
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        # Top and bottom edges
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        # Left and right edges
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        # Corners
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_glass_panel(frame, x1, y1, x2, y2, border_color=None, alpha=0.6, radius=10):
    """Draw a glassmorphism-style dark panel."""
    # Fill background
    alpha_blend(frame, COLORS["bg_card"], alpha, (x1, y1, x2, y2))
    if border_color:
        draw_rounded_rect(frame, x1, y1, x2, y2, border_color, thickness=1, radius=radius)


def draw_corner_brackets(frame, x, y, w, h, color, size=20, thickness=3):
    """Draw corner bracket decorations around a face box."""
    # Top-left
    cv2.line(frame, (x, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + size), color, thickness)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - size, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + size), color, thickness)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + size, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - size), color, thickness)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - size, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - size), color, thickness)


def draw_scanning_line(frame, x, y, w, h, color, t):
    """Draw an animated scanning line over the face region."""
    period = 2.0  # seconds per cycle
    phase = (t % period) / period
    scan_y = int(y + h * phase)
    scan_y = min(scan_y, y + h - 2)
    # Gradient line
    for i in range(max(0, scan_y - 3), min(frame.shape[0], scan_y + 3)):
        alpha_val = 1.0 - abs(i - scan_y) / 4.0
        overlay_color = tuple(int(c * alpha_val) for c in color)
        cv2.line(frame, (x, i), (x + w, i), overlay_color, 1)


def put_text_with_shadow(frame, text, pos, font, scale, color, thickness=1, shadow_offset=1):
    """Draw text with a subtle drop shadow."""
    sx, sy = pos[0] + shadow_offset, pos[1] + shadow_offset
    cv2.putText(frame, text, (sx, sy), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


class UIRenderer:
    """Handles all UI drawing for the Smart Attendance System."""

    def __init__(self, frame_w, frame_h):
        self.fw = frame_w
        self.fh = frame_h
        self.start_time = time.time()
        self.pulse_phase = 0.0

        # Panel dimensions
        self.header_h = 55
        self.right_panel_w = 230
        self.bottom_bar_h = 40
        self.log_panel_h = 160

    def _now(self):
        return time.time() - self.start_time

    def draw_header(self, frame, fps, total_today, tracking_count):
        """Draw the top header bar."""
        h_h = self.header_h
        # Background
        alpha_blend(frame, COLORS["overlay_bg"], 0.88, (0, 0, self.fw, h_h))
        # Left gradient accent line
        for x in range(self.fw):
            t = x / self.fw
            b = int(255 * (1 - t) * 0.8)
            g = int(212 * t * 0.6)
            r = int(99 * t)
            cv2.line(frame, (x, h_h - 2), (x, h_h - 1), (b, g, r), 1)

        # Logo / title
        cv2.putText(frame, "SMART", (12, 28), FONT_BOLD, 0.7, COLORS["accent"], 2, cv2.LINE_AA)
        cv2.putText(frame, "ATTENDANCE", (75, 28), FONT_BOLD, 0.7, COLORS["white"], 1, cv2.LINE_AA)
        cv2.putText(frame, "SYSTEM", (230, 28), FONT_BOLD, 0.7, COLORS["accent2"], 1, cv2.LINE_AA)

        # Subtitle
        now_str = datetime.now().strftime("%A, %d %b %Y  |  %H:%M:%S")
        cv2.putText(frame, now_str, (12, 46), FONT, 0.38, COLORS["gray_mid"], 1, cv2.LINE_AA)

        # Right stats
        # FPS badge
        fps_text = f"FPS {fps:4.1f}"
        fps_color = COLORS["success"] if fps >= 20 else COLORS["warning"]
        cv2.putText(frame, fps_text, (self.fw - self.right_panel_w - 5, 22), FONT_BOLD, 0.5, fps_color, 1, cv2.LINE_AA)

        # Faces detected
        faces_text = f"FACES {tracking_count:02d}"
        cv2.putText(frame, faces_text, (self.fw - self.right_panel_w - 5, 42), FONT, 0.4, COLORS["accent"], 1, cv2.LINE_AA)

        # Present count badge
        badge_x = self.fw - 80
        badge_y = 8
        alpha_blend(frame, COLORS["accent3"], 0.3, (badge_x, badge_y, self.fw - 8, badge_y + 40))
        draw_rounded_rect(frame, badge_x, badge_y, self.fw - 8, badge_y + 40, COLORS["accent3"], radius=6)
        cv2.putText(frame, f"{total_today:02d}", (badge_x + 10, badge_y + 28), FONT_BOLD, 1.0, COLORS["white"], 2, cv2.LINE_AA)
        cv2.putText(frame, "TODAY", (badge_x + 10, badge_y + 38), FONT, 0.3, COLORS["accent3"], 1, cv2.LINE_AA)

    def draw_right_panel(self, frame, gesture_on, liveness_on, preproc_summary, stats, registered_users):
        """Draw the right-side status and stats panel."""
        px = self.fw - self.right_panel_w + 4
        py = self.header_h + 5
        pw = self.right_panel_w - 8
        panel_h = self.fh - self.header_h - self.bottom_bar_h - 10

        # Panel background
        draw_glass_panel(frame, px, py, px + pw, py + panel_h,
                         border_color=COLORS["gray_dark"], alpha=0.75, radius=8)

        cy = py + 18
        section_w = pw - 10

        # ── STATUS section ──────────────────────────────────────────
        put_text_with_shadow(frame, "STATUS", (px + 8, cy), FONT_BOLD, 0.42, COLORS["accent"], 1)
        cy += 4
        cv2.line(frame, (px + 8, cy), (px + pw - 8, cy), COLORS["gray_dark"], 1)
        cy += 14

        def status_row(label, on_off, on_label="ON", off_label="OFF"):
            nonlocal cy
            cv2.putText(frame, label, (px + 8, cy), FONT, 0.38, COLORS["gray_light"], 1, cv2.LINE_AA)
            col = COLORS["success"] if on_off else COLORS["danger"]
            text = on_label if on_off else off_label
            cv2.putText(frame, text, (px + pw - 42, cy), FONT_BOLD, 0.38, col, 1, cv2.LINE_AA)
            cy += 18

        status_row("Gesture", gesture_on)
        status_row("Liveness", liveness_on)
        status_row("Tracking", True)
        status_row("Recognition", stats.get("total_all_time", 0) > 0)

        cy += 6

        # ── STATISTICS section ──────────────────────────────────────
        put_text_with_shadow(frame, "STATISTICS", (px + 8, cy), FONT_BOLD, 0.42, COLORS["accent"], 1)
        cy += 4
        cv2.line(frame, (px + 8, cy), (px + pw - 8, cy), COLORS["gray_dark"], 1)
        cy += 14

        def stat_row(label, value, value_color=None):
            nonlocal cy
            cv2.putText(frame, label, (px + 8, cy), FONT, 0.36, COLORS["gray_light"], 1, cv2.LINE_AA)
            vc = value_color or COLORS["white"]
            cv2.putText(frame, str(value), (px + pw - 40, cy), FONT_BOLD, 0.4, vc, 1, cv2.LINE_AA)
            cy += 18

        stat_row("Present Today", stats.get("total_today", 0), COLORS["accent3"])
        stat_row("All-Time Logs", stats.get("total_all_time", 0), COLORS["accent"])
        stat_row("Registered", len(registered_users), COLORS["accent2"])
        stat_row("Last Saved", stats.get("last_save", "--:--"), COLORS["gray_mid"])

        cy += 6

        # ── PREPROCESSING section ────────────────────────────────────
        put_text_with_shadow(frame, "PIPELINE", (px + 8, cy), FONT_BOLD, 0.42, COLORS["accent"], 1)
        cy += 4
        cv2.line(frame, (px + 8, cy), (px + pw - 8, cy), COLORS["gray_dark"], 1)
        cy += 12

        parts = preproc_summary.split(",") if preproc_summary else []
        for part in parts[:5]:
            part = part.strip()
            if part:
                dot_col = COLORS["success"] if "ON" in part else COLORS["gray_dark"]
                cv2.circle(frame, (px + 12, cy - 4), 4, dot_col, -1)
                cv2.putText(frame, part, (px + 22, cy), FONT, 0.32, COLORS["gray_light"], 1, cv2.LINE_AA)
                cy += 14

        cy += 6

        # ── REGISTERED USERS ────────────────────────────────────────
        if cy + 60 < py + panel_h:
            put_text_with_shadow(frame, "REGISTERED", (px + 8, cy), FONT_BOLD, 0.42, COLORS["accent"], 1)
            cy += 4
            cv2.line(frame, (px + 8, cy), (px + pw - 8, cy), COLORS["gray_dark"], 1)
            cy += 12

            for name, uid in registered_users[:5]:
                cv2.circle(frame, (px + 12, cy - 4), 4, COLORS["accent2"], -1)
                display = name[:16] + ".." if len(name) > 16 else name
                cv2.putText(frame, display, (px + 22, cy), FONT, 0.32, COLORS["white"], 1, cv2.LINE_AA)
                cy += 14

            if len(registered_users) > 5:
                cv2.putText(frame, f"  ...+{len(registered_users)-5} more",
                            (px + 8, cy), FONT, 0.32, COLORS["gray_mid"], 1, cv2.LINE_AA)

    def draw_attendance_log(self, frame, records):
        """Draw a recent-attendance log panel at the bottom-left."""
        lw = 300
        lh = self.log_panel_h
        lx = 8
        ly = self.fh - self.bottom_bar_h - lh - 8

        draw_glass_panel(frame, lx, ly, lx + lw, ly + lh,
                         border_color=COLORS["gray_dark"], alpha=0.78, radius=8)

        put_text_with_shadow(frame, "RECENT ATTENDANCE", (lx + 10, ly + 16),
                             FONT_BOLD, 0.42, COLORS["accent3"], 1)
        cv2.line(frame, (lx + 8, ly + 20), (lx + lw - 8, ly + 20), COLORS["gray_dark"], 1)

        y = ly + 36
        if not records:
            cv2.putText(frame, "No attendance logged yet", (lx + 10, y),
                        FONT, 0.38, COLORS["gray_mid"], 1, cv2.LINE_AA)
        else:
            items = list(records.items())[-6:]
            for name, record in items:
                time_str = record["first_seen"].strftime("%H:%M:%S")
                method = record.get("method", "auto").upper()[:3]
                entries = record.get("entries", 1)

                # Row background
                row_col = COLORS["success"] if record.get("status") == "PRESENT" else COLORS["gray_dark"]
                cv2.circle(frame, (lx + 14, y - 4), 4, row_col, -1)

                name_disp = name[:14] + ".." if len(name) > 14 else name
                cv2.putText(frame, name_disp, (lx + 24, y), FONT_BOLD, 0.38, COLORS["white"], 1, cv2.LINE_AA)
                cv2.putText(frame, time_str, (lx + 165, y), FONT, 0.35, COLORS["gray_light"], 1, cv2.LINE_AA)
                cv2.putText(frame, f"{method} x{entries}", (lx + 240, y), FONT, 0.3, COLORS["accent"], 1, cv2.LINE_AA)
                y += 19

    def draw_face_overlay(self, frame, face, name, confidence, is_live, object_id, marked, t):
        """Draw simple face detection overlay with box and info tag."""
        x, y, w, h = face["bbox"]

        if name == "Unknown":
            bracket_color = COLORS["face_unknown"]
            label_bg = COLORS["face_unknown"]
        else:
            bracket_color = COLORS["face_known"] if is_live else COLORS["warning"]
            label_bg = COLORS["face_known"] if is_live else COLORS["warning"]

        # Simple bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), bracket_color, 2)

        # Name label
        label = f" {name} " if name != "Unknown" else " UNKNOWN "
        (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.6, 1)
        tag_y = max(0, y - th - 10)
        
        cv2.rectangle(frame, (x, tag_y), (x + tw, tag_y + th + 10), label_bg, -1)
        cv2.putText(frame, label, (x, tag_y + th + 5), FONT_BOLD, 0.6, COLORS["white"], 1, cv2.LINE_AA)

    def draw_notification(self, frame, messages, current_time, duration):
        """Draw floating notification toasts."""
        active = [(msg, ts) for msg, ts in messages if current_time - ts < duration]
        if not active:
            return

        nx = 10
        ny_base = self.header_h + 10
        for i, (msg, ts) in enumerate(active[-3:]):
            age = current_time - ts
            alpha_val = max(0.0, 1.0 - (age / duration) * 0.8)
            nh = 30
            ny = ny_base + i * (nh + 6)

            (tw, _), _ = cv2.getTextSize(msg, FONT, 0.5, 1)
            toast_w = tw + 40
            alpha_blend(frame, COLORS["bg_card2"], 0.85, (nx, ny, nx + toast_w, ny + nh))
            cv2.rectangle(frame, (nx, ny), (nx + toast_w, ny + nh), COLORS["accent3"], 1)
            # Left accent bar
            cv2.rectangle(frame, (nx, ny), (nx + 3, ny + nh), COLORS["accent3"], -1)
            put_text_with_shadow(frame, "✓ " + msg, (nx + 10, ny + 20),
                                 FONT, 0.5, COLORS["white"], 1)

    def draw_bottom_bar(self, frame):
        """Draw the bottom controls help bar."""
        by = self.fh - self.bottom_bar_h
        alpha_blend(frame, COLORS["overlay_bg"], 0.88, (0, by, self.fw, self.fh))
        cv2.line(frame, (0, by), (self.fw, by), COLORS["gray_dark"], 1)

        keys = [
            ("Q", "Quit"),
            ("S", "Save Report"),
            ("R", "Register Face"),
            ("G", "Toggle Gesture"),
            ("L", "Toggle Liveness"),
            ("D", "Dashboard"),
        ]
        x = 10
        for key, label in keys:
            # Key badge
            kw = 16
            cv2.rectangle(frame, (x, by + 8), (x + kw, by + 28), COLORS["gray_dark"], -1)
            cv2.rectangle(frame, (x, by + 8), (x + kw, by + 28), COLORS["gray_mid"], 1)
            cv2.putText(frame, key, (x + 4, by + 23), FONT_BOLD, 0.38, COLORS["white"], 1, cv2.LINE_AA)
            x += kw + 3
            cv2.putText(frame, label, (x, by + 23), FONT, 0.38, COLORS["gray_light"], 1, cv2.LINE_AA)
            (lw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            x += lw + 15

    def draw_register_overlay(self, frame, name_so_far):
        """Draw the registration mode overlay."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.fw, self.fh), COLORS["overlay_bg"], -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cx = self.fw // 2
        cy = self.fh // 2

        # Main card
        cw, ch = 420, 160
        draw_glass_panel(frame, cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2,
                         border_color=COLORS["accent2"], alpha=0.85, radius=12)
        draw_rounded_rect(frame, cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2,
                          COLORS["accent2"], thickness=2, radius=12)

        put_text_with_shadow(frame, "REGISTER NEW FACE",
                             (cx - 130, cy - 45), FONT_BOLD, 0.8, COLORS["accent2"], 2)

        # Name field
        field_y = cy - 10
        cv2.rectangle(frame, (cx - 160, field_y - 20), (cx + 160, field_y + 10),
                      COLORS["gray_dark"], -1)
        cv2.rectangle(frame, (cx - 160, field_y - 20), (cx + 160, field_y + 10),
                      COLORS["accent"], 1)
        display_name = name_so_far if name_so_far else "Enter name..."
        name_col = COLORS["white"] if name_so_far else COLORS["gray_mid"]
        cv2.putText(frame, display_name, (cx - 150, field_y + 5), FONT, 0.55, name_col, 1, cv2.LINE_AA)

        cv2.putText(frame, "ENTER = Confirm    ESC = Cancel",
                    (cx - 145, cy + 50), FONT, 0.43, COLORS["gray_light"], 1, cv2.LINE_AA)

    def draw_frame(self, frame, context: dict) -> np.ndarray:
        """
        Master draw call. Applies minimalistic UI elements on top of the given frame.
        """
        t = self._now()

        # 1. Draw face overlays (below panels)
        faces = context.get("faces", [])
        tracked_objects = context.get("tracked_objects", [])
        for face, (oid, centroid, name, is_new, marked) in zip(faces, tracked_objects):
            conf = face.get("confidence", 0.0)
            is_live = face.get("is_live", True)
            self.draw_face_overlay(frame, face, name, conf, is_live, oid, marked, t)

        # 2. Notifications / toasts (Keep this so they know attendance is marked)
        self.draw_notification(
            frame,
            context.get("log_messages", []),
            context.get("current_time", time.time()),
            context.get("log_duration", 3)
        )

        # 3. Register overlay (if active)
        if context.get("register_mode", False):
            self.draw_register_overlay(frame, context.get("register_name", ""))

        return frame
