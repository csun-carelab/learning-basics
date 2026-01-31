#!/usr/bin/env python

"""
eFlesh visualization + recording + simple force calibration (pound-force, lbf).

Key bindings
- B : re-compute baseline (average of a few samples)
- R : toggle recording (saves baseline-subtracted ΔB data on stop/exit)
- C : capture a calibration point (prompts you to enter known force in lbf)
- F : fit calibration model from captured points (linear least-squares)
- L : load calibration model (json)
- S : save calibration model (json)
- ESC / window close : quit (auto-saves if recording was on)

What "calibration" means here:
We learn a linear model mapping baseline-subtracted magnetometer deltas (ΔBx,ΔBy,ΔBz)
to force components (Fx,Fy,Fz) in pound-force (lbf):

    [Fx Fy Fz] = [ΔBx ΔBy ΔBz 1] @ A

where A is a 4x3 matrix learned from your calibration points.
"""

import time
import numpy as np
import torch
import os
import json
from typing import Optional, Tuple, List

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import sys
import pygame
from datetime import datetime
from anyskin import AnySkinProcess
import argparse


# Configuration constants
CHIP_LOCATIONS = np.array([[455, 453], [275, 451], [624, 455], [451, 292], [454, 613]])
CHIP_XY_ROTATIONS = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0])
NO_CONTACT_THRESHOLD = 200  # ΔB magnitude threshold for no-contact detection


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _default_save_dir() -> str:
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    os.makedirs(d, exist_ok=True)
    return d


def load_calibration(calib_path: str) -> Optional[np.ndarray]:
    """Load calibration matrix from JSON file."""
    if not calib_path or not os.path.exists(calib_path):
        return None
    try:
        with open(calib_path, "r") as f:
            obj = json.load(f)
        A = np.array(obj.get("A", None), dtype=float)
        if A.shape != (4, 3):
            raise ValueError(f"Calibration file has wrong shape: {A.shape}, expected (4,3)")
        return A
    except Exception as e:
        print(f"[calib] Error loading calibration: {e}")
        return None


def save_calibration(calib_path: str, A: np.ndarray, meta: Optional[dict] = None) -> None:
    """Save calibration matrix to JSON file."""
    try:
        os.makedirs(os.path.dirname(calib_path) or ".", exist_ok=True)
        payload = {"A": A.tolist(), "meta": meta or {}, "saved_at": _now_tag()}
        with open(calib_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[calib] Saved calibration to: {calib_path}")
    except Exception as e:
        print(f"[calib] Error saving calibration: {e}")


def fit_linear_calibration(X_dB: np.ndarray, Y_F: np.ndarray) -> np.ndarray:
    """
    Fit linear calibration model using least squares.
    
    X_dB: (N, 3) baseline-subtracted deltas
    Y_F : (N, 3) known forces in pound-force (lbf)
    returns A: (4, 3) such that [X 1] @ A ~= Y
    """
    if X_dB.ndim != 2 or X_dB.shape[1] != 3:
        raise ValueError("X_dB must be (N,3)")
    if Y_F.ndim != 2 or Y_F.shape[1] != 3:
        raise ValueError("Y_F must be (N,3)")
    if X_dB.shape[0] < 4:
        raise ValueError("Need at least 4 calibration points (recommended 8+).")

    X_aug = np.concatenate([X_dB, np.ones((X_dB.shape[0], 1))], axis=1)  # (N,4)
    # Least squares: X_aug @ A = Y
    A, *_ = np.linalg.lstsq(X_aug, Y_F, rcond=None)
    # A shape: (4,3)
    return A


class CalibrationInputBox:
    """Non-blocking text input box for calibration force entry."""
    
    def __init__(self, x: int, y: int, width: int, height: int, font: pygame.font.Font):
        self.rect = pygame.Rect(x, y, width, height)
        self.color_inactive = pygame.Color(100, 100, 100)
        self.color_active = pygame.Color(50, 150, 50)
        self.color = self.color_inactive
        self.text = ""
        self.font = font
        self.active = False
        self.prompt = "Enter force (Fz or Fx,Fy,Fz): "
        
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Handle input events. Returns input text if Enter pressed, None otherwise."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                self.color = self.color_active
            else:
                self.active = False
                self.color = self.color_inactive
                
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                result = self.text
                self.text = ""
                self.active = False
                self.color = self.color_inactive
                return result
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.text = ""
                self.active = False
                self.color = self.color_inactive
            else:
                # Only accept valid characters for force input
                if event.unicode in "0123456789.,-+":
                    self.text += event.unicode
        return None
    
    def draw(self, surface: pygame.Surface) -> None:
        """Draw the input box if active."""
        if self.active:
            # Draw semi-transparent background
            bg = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 128))
            surface.blit(bg, (0, 0))
            
            # Draw input box
            pygame.draw.rect(surface, self.color, self.rect, 2)
            pygame.draw.rect(surface, (240, 240, 240), self.rect)
            pygame.draw.rect(surface, self.color, self.rect, 2)
            
            # Draw prompt
            prompt_surf = self.font.render(self.prompt, True, (0, 0, 0))
            surface.blit(prompt_surf, (self.rect.x + 5, self.rect.y - 30))
            
            # Draw text
            txt_surface = self.font.render(self.text, True, (0, 0, 0))
            surface.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))


def visualize(
    port: str,
    file: Optional[str] = None,
    viz_mode: str = "3axis",
    scaling: float = 10.0,
    record: bool = False,
    calib_path: Optional[str] = None,
    force_viz_scale: float = 0.25,  # pixels per lbf (only used when calibrated)
):
    # Initialize sensor stream or load file
    sensor_stream = None
    load_data = None
    
    if file is None:
        try:
            sensor_stream = AnySkinProcess(num_mags=5, port=port)
            sensor_stream.start()
            time.sleep(1.0)
        except Exception as e:
            print(f"[ERROR] Failed to initialize sensor: {e}")
            return
    else:
        try:
            load_data = np.loadtxt(file)
        except Exception as e:
            print(f"[ERROR] Failed to load file: {e}")
            return

    pygame.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bg_image_path = os.path.join(dir_path, "flesh.png")
    
    try:
        bg_image = pygame.image.load(bg_image_path)
    except Exception as e:
        print(f"[WARNING] Could not load background image: {e}")
        # Create a blank background
        bg_image = pygame.Surface((900, 900))
        bg_image.fill((234, 237, 232))

    image_width, image_height = bg_image.get_size()
    aspect_ratio = image_height / image_width
    desired_width = 900
    desired_height = int(desired_width * aspect_ratio)

    bg_image = pygame.transform.scale(bg_image, (desired_width, desired_height))
    window = pygame.display.set_mode((desired_width, desired_height), pygame.SRCALPHA)
    background_surface = pygame.Surface(window.get_size(), pygame.SRCALPHA)
    background_surface.fill((234, 237, 232, 255))
    background_surface.blit(bg_image, (0, 0))
    pygame.display.set_caption("eFlesh Visualization (Calibrated Force Option)")

    # --- State ---
    data_buffer: List[np.ndarray] = []  # baseline-subtracted samples for recording
    is_recording = False
    record_filename_base: Optional[str] = None

    # Calibration state
    calib_X: List[np.ndarray] = []  # list of (3,) ΔB_mean
    calib_Y: List[np.ndarray] = []  # list of (3,) F in lbf
    A: Optional[np.ndarray] = None
    
    # Calibration capture state
    awaiting_force_input = False
    captured_deltaB_mean: Optional[np.ndarray] = None
    
    if calib_path:
        A = load_calibration(calib_path)
        if A is not None:
            print(f"[calib] Loaded calibration from: {calib_path}")

    f_big = pygame.font.Font(None, 42)
    f_small = pygame.font.Font(None, 28)
    
    # Create input box for calibration
    input_box = CalibrationInputBox(
        desired_width // 2 - 200, 
        desired_height // 2 - 20, 
        400, 
        40, 
        f_small
    )

    def get_baseline(num_samples: int = 10) -> Optional[np.ndarray]:
        """Compute baseline from sensor data."""
        try:
            baseline_data = sensor_stream.get_data(num_samples=num_samples)
            baseline_data = np.array(baseline_data)[:, 1:]
            return np.mean(baseline_data, axis=0)
        except Exception as e:
            print(f"[ERROR] Failed to get baseline: {e}")
            return None

    def deltaB_to_force(deltaB_15: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Convert deltaB to force estimate if calibrated.
        
        deltaB_15: (15,) = xyzxyz... (5 mags)
        Returns:
          Fxyz_est: (3,) in pound-force (lbf) if calibrated, else None
          deltaB_mean: (3,) mean across mags
        """
        d = deltaB_15.reshape(5, 3)
        d[:, :2] *= -1  # match axis convention (flip x,y)
        deltaB_mean = d.mean(axis=0)  # (3,)
        
        if A is None:
            return None, deltaB_mean
            
        x_aug = np.array([deltaB_mean[0], deltaB_mean[1], deltaB_mean[2], 1.0], dtype=float)
        F = x_aug @ A  # (3,)
        return F, deltaB_mean

    def draw_overlay(is_calibrated: bool, F: Optional[np.ndarray], n_calib: int) -> None:
        """Draw status overlay on screen."""
        y = 10
        if is_recording:
            t = f_big.render("REC", True, (200, 0, 0))
            window.blit(t, (10, y))
            y += 40

        if is_calibrated:
            t = f_small.render("Calibrated: ON (lbf)", True, (0, 120, 0))
            window.blit(t, (10, y))
            y += 28
        else:
            t = f_small.render("Calibrated: OFF (showing ΔB units)", True, (120, 0, 0))
            window.blit(t, (10, y))
            y += 28

        t = f_small.render(f"Calib points: {n_calib}  (C=capture, F=fit, S=save, L=load)", True, (30, 30, 30))
        window.blit(t, (10, y))
        y += 28

        if F is not None:
            fx, fy, fz = F
            mag = float(np.linalg.norm(F))
            window.blit(f_small.render(f"Fx={fx:+.2f} lbf  Fy={fy:+.2f} lbf  Fz={fz:+.2f} lbf", True, (10, 10, 10)), (10, y))
            y += 26
            window.blit(f_small.render(f"|F|={mag:.2f} lbf", True, (10, 10, 10)), (10, y))

    def visualize_data(deltaB_15: np.ndarray) -> None:
        """
        Visualize sensor data.
        - Z (or Fz if calibrated): red circle radius
        - XY (or Fx,Fy if calibrated): green arrow direction
        """
        d = deltaB_15.reshape(5, 3)
        d[:, :2] *= -1  # Flip x and y axes

        F_est, deltaB_mean = deltaB_to_force(deltaB_15)
        is_cal = F_est is not None

        for magid, chip_location in enumerate(CHIP_LOCATIONS):
            if viz_mode == "magnitude":
                data_mag = float(np.linalg.norm(d[magid]))
                pygame.draw.circle(window, (255, 83, 72), chip_location, int(data_mag / scaling))

            elif viz_mode == "3axis":
                # No-contact detection
                flat = d.flatten()
                if np.linalg.norm(flat) < NO_CONTACT_THRESHOLD:
                    t = f_big.render("No Contact", True, (200, 0, 0))
                    window.blit(t, (30, desired_height - 60))

                # Circle radius from Z-component
                if is_cal:
                    # Use estimated normal force magnitude for lbf-based scale
                    z_radius = abs(float(F_est[2])) / max(1e-9, force_viz_scale)
                else:
                    z_radius = abs(float(d[magid, 2])) / scaling

                # Thickness cue based on sign of z
                width = 2 if (d[magid, 2] < 0) else 0
                pygame.draw.circle(window, (255, 0, 0), tuple(chip_location), int(z_radius), width)

                # Arrow from XY
                rotation_mat = np.array([
                    [np.cos(CHIP_XY_ROTATIONS[magid]), -np.sin(CHIP_XY_ROTATIONS[magid])],
                    [np.sin(CHIP_XY_ROTATIONS[magid]),  np.cos(CHIP_XY_ROTATIONS[magid])],
                ])
                data_xy = rotation_mat @ d[magid, :2]

                if is_cal:
                    # Use calibrated shear from global model
                    shear = np.array([F_est[0], F_est[1]], dtype=float)
                    arrow_vec = shear / max(1e-9, force_viz_scale)
                else:
                    arrow_vec = data_xy / scaling

                arrow_end = (
                    int(chip_location[0] + arrow_vec[0]), 
                    int(chip_location[1] + arrow_vec[1])
                )
                pygame.draw.line(window, (0, 255, 0), tuple(chip_location), arrow_end, 8)

        # Overlay numeric force
        draw_overlay(is_cal, F_est, len(calib_X))

    # --- Baseline ---
    baseline = np.zeros(15, dtype=float)
    if file is None:
        baseline_result = get_baseline(num_samples=10)
        if baseline_result is not None:
            baseline = baseline_result

    # --- Playback vs live ---
    playback_idx = 0

    # --- Main loop ---
    running = True
    clock = pygame.time.Clock()
    FPS = 60

    # Default paths
    save_dir = _default_save_dir()
    if calib_path is None:
        calib_path = os.path.join(save_dir, "eflesh_calibration.json")

    while running:
        window.blit(background_surface, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                print(f"Mouse clicked at ({x}, {y})")

            # Handle input box events
            if awaiting_force_input:
                force_input = input_box.handle_event(event)
                if force_input is not None:
                    # Process the calibration input
                    try:
                        if "," in force_input:
                            parts = [float(x) for x in force_input.split(",")]
                            if len(parts) != 3:
                                raise ValueError("Need 3 values: Fx,Fy,Fz")
                            F_known = np.array(parts, dtype=float)
                        else:
                            # Assume normal force only
                            F_known = np.array([0.0, 0.0, float(force_input)], dtype=float)
                        
                        calib_X.append(captured_deltaB_mean)
                        calib_Y.append(F_known)
                        print(f"[calib] Added point #{len(calib_X)}: ΔB_mean={captured_deltaB_mean} -> F={F_known} lbf")
                        awaiting_force_input = False
                        captured_deltaB_mean = None
                    except Exception as e:
                        print(f"[calib] Invalid force input: {e}. Point not added.")
                        awaiting_force_input = False
                        captured_deltaB_mean = None

            if event.type == pygame.KEYDOWN:
                # Don't process other keys if input box is active
                if input_box.active:
                    continue
                    
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Baseline reset
                if event.key == pygame.K_b and file is None:
                    baseline_result = get_baseline(num_samples=10)
                    if baseline_result is not None:
                        baseline = baseline_result
                        print("[baseline] Updated.")

                # Recording toggle
                if event.key == pygame.K_r:
                    if not is_recording:
                        is_recording = True
                        record_filename_base = os.path.join(save_dir, f"eflesh_{_now_tag()}")
                        data_buffer = []
                        print(f"[record] START -> {record_filename_base}(.pt/.txt)")
                    else:
                        is_recording = False
                        print("[record] STOP -> saving...")
                        # Save now
                        arr = np.array(data_buffer, dtype=float) if len(data_buffer) else np.zeros((0, 15))
                        try:
                            torch.save(torch.tensor(arr, dtype=torch.float32), record_filename_base + ".pt")
                            np.savetxt(record_filename_base + ".txt", arr)
                            print(f"[record] Saved {len(arr)} samples.")
                        except Exception as e:
                            print(f"[record] Error saving: {e}")

                # Capture calibration point
                if event.key == pygame.K_c and file is None and not awaiting_force_input:
                    try:
                        # Average a window for stability
                        samples = sensor_stream.get_data(num_samples=25)
                        samples = np.array(samples)[:, 1:]
                        dB = samples - baseline  # (N,15)
                        dB_reshaped = dB.mean(axis=0).reshape(5, 3)
                        dB_reshaped[:, :2] *= -1
                        captured_deltaB_mean = dB_reshaped.mean(axis=0)  # (3,)
                        print(f"[calib] Captured ΔB_mean = {captured_deltaB_mean}")
                        
                        # Activate input box
                        awaiting_force_input = True
                        input_box.active = True
                        input_box.color = input_box.color_active
                    except Exception as e:
                        print(f"[calib] Error capturing calibration point: {e}")

                # Fit calibration model
                if event.key == pygame.K_f:
                    try:
                        X = np.array(calib_X, dtype=float)
                        Y = np.array(calib_Y, dtype=float)
                        A = fit_linear_calibration(X, Y)
                        print("[calib] Fit complete. A=")
                        print(A)
                    except Exception as e:
                        print(f"[calib] Fit failed: {e}")

                # Save calibration model
                if event.key == pygame.K_s:
                    if A is None:
                        print("[calib] Nothing to save (fit or load first).")
                    else:
                        meta = {
                            "note": "Linear ΔB_mean->F model. F=[ΔBx,ΔBy,ΔBz,1]@A", 
                            "n_points": len(calib_X)
                        }
                        save_calibration(calib_path, A, meta=meta)

                # Load calibration model
                if event.key == pygame.K_l:
                    loaded_A = load_calibration(calib_path)
                    if loaded_A is not None:
                        A = loaded_A
                        print(f"[calib] Loaded: {calib_path}")

        # Get data
        deltaB = np.zeros(15, dtype=float)
        
        if file is not None:
            # Playback mode
            if playback_idx >= len(load_data):
                playback_idx = 0
            sensor_data = load_data[playback_idx]
            playback_idx += 1
            baseline_play = np.zeros_like(sensor_data)
            deltaB = sensor_data - baseline_play
        else:
            # Live mode
            try:
                sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
                deltaB = sensor_data - baseline
            except Exception as e:
                print(f"[ERROR] Failed to get sensor data: {e}")

        # Buffer for recording
        if is_recording:
            data_buffer.append(deltaB.copy())

        # Draw visualization
        visualize_data(deltaB)
        
        # Draw input box if active
        if awaiting_force_input:
            input_box.draw(window)

        pygame.display.update()
        clock.tick(FPS)

    # --- Cleanup ---
    pygame.quit()
    
    if sensor_stream is not None:
        try:
            sensor_stream.pause_streaming()
            sensor_stream.join()
        except Exception as e:
            print(f"[WARNING] Error during sensor cleanup: {e}")

    # Auto-save if quitting while recording
    if is_recording and record_filename_base:
        arr = np.array(data_buffer, dtype=float) if len(data_buffer) else np.zeros((0, 15))
        try:
            torch.save(torch.tensor(arr, dtype=torch.float32), record_filename_base + ".pt")
            np.savetxt(record_filename_base + ".txt", arr)
            print(f"[record] Auto-saved {len(arr)} samples on exit -> {record_filename_base}(.pt/.txt)")
        except Exception as e:
            print(f"[record] Error auto-saving: {e}")


def default_viz(argv=sys.argv):
    visualize(port=argv[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eFlesh viz + recording + linear force calibration (lbf).")
    parser.add_argument("-p", "--port", type=str, help="Serial port (e.g., COM3)", default="COM3")
    parser.add_argument("-f", "--file", type=str, help="path to load data from", default=None)
    parser.add_argument("-v", "--viz_mode", type=str, help="visualization mode", default="3axis", choices=["magnitude", "3axis"])
    parser.add_argument("-s", "--scaling", type=float, help="ΔB scaling factor for visualization (when not calibrated)", default=10.0)
    parser.add_argument("-r", "--record", action="store_true", help="(kept for compatibility) start with recording OFF; use R to toggle")
    parser.add_argument("--calib_path", type=str, default=None, help="Path to calibration json (default: ./data/eflesh_calibration.json)")
    parser.add_argument("--force_viz_scale", type=float, default=0.25, help="Pixels per lbf when calibrated (bigger=larger visuals).")
    args = parser.parse_args()

    visualize(
        port=args.port,
        file=args.file,
        viz_mode=args.viz_mode,
        scaling=args.scaling,
        record=args.record,
        calib_path=args.calib_path,
        force_viz_scale=args.force_viz_scale,
    )
