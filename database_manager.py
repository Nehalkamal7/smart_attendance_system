"""
Database Manager Module
Addresses Rubric Section 7: Output / Result Display
- Save result to CSV/Excel or generate report
- Real-time attendance log
"""
import os
import csv
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
from config import REPORTS_DIR, REPORT, TRACKING


class AttendanceManager:
    """Manages attendance records and report generation."""

    def __init__(self):
        self.records = OrderedDict()  # person_name -> {timestamp, status, id}
        self.csv_path = os.path.join(REPORTS_DIR, REPORT["csv_filename"])
        self.excel_path = os.path.join(REPORTS_DIR, REPORT["excel_filename"])
        self.master_csv_path = os.path.join(REPORTS_DIR, "master_attendance_log.csv")
        self.last_save = datetime.now()
        self.auto_save_interval = REPORT["auto_save_interval"]
        self.attendance_cooldown = {}  # person_name -> last_marked_time
        self.cooldown_seconds = TRACKING["attendance_cooldown"]

        # Load existing records if any
        self.load_csv()
        
        # Initial sync to master log for history if it's new
        self._sync_to_master_log()

    def mark_attendance(self, name, person_id=None, confidence=0.0, method="auto"):
        """
        Mark attendance for a person.

        Args:
            name: Person's name
            person_id: Optional ID
            confidence: Recognition confidence
            method: "auto" (automatic) or "gesture" (wave confirmed)

        Returns:
            (success: bool, message: str, is_new: bool)
        """
        now = datetime.now()

        # Check cooldown
        if name in self.attendance_cooldown:
            time_diff = (now - self.attendance_cooldown[name]).total_seconds()
            if time_diff < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - time_diff)
                return False, f"Cooldown: {remaining}s remaining", False

        # Check if already marked today
        is_new = name not in self.records

        if is_new:
            self.records[name] = {
                "name": name,
                "id": person_id or "N/A",
                "first_seen": now,
                "last_seen": now,
                "status": "PRESENT",
                "confidence": confidence,
                "method": method,
                "entries": 1
            }
            # Append to master log for history (one entry per day)
            self._append_to_master_log(self.records[name])
            message = f"Attendance marked for {name}"
        else:
            self.records[name]["last_seen"] = now
            self.records[name]["entries"] += 1
            self.records[name]["confidence"] = max(self.records[name]["confidence"], confidence)
            message = f"Updated attendance for {name}"

        self.attendance_cooldown[name] = now

        # Auto-save check
        if (now - self.last_save).total_seconds() > self.auto_save_interval:
            self.save_csv()
            self.save_excel()

        return True, message, is_new

    def get_records(self):
        """Return all attendance records."""
        return self.records

    def get_today_records(self):
        """Return records for today only."""
        today = datetime.now().date()
        return {k: v for k, v in self.records.items() 
                if v["first_seen"].date() == today}

    def save_csv(self):
        """Save attendance records to CSV file."""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "ID", "First Seen", "Last Seen", 
                               "Status", "Confidence", "Method", "Entries"])

                for record in self.records.values():
                    writer.writerow([
                        record["name"],
                        record["id"],
                        record["first_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        record["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                        record["status"],
                        f"{record['confidence']:.2f}",
                        record["method"],
                        record["entries"]
                    ])

            self.last_save = datetime.now()
            return True, f"Saved to {self.csv_path}"
        except Exception as e:
            return False, f"Error saving CSV: {e}"

    def load_csv(self):
        """Load existing attendance records from CSV. Only loads today's records."""
        if not os.path.exists(self.csv_path):
            return

        today = datetime.now().date()
        old_count = 0
        
        try:
            temp_records = OrderedDict()
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row["Name"]
                    try:
                        first_seen = datetime.strptime(row["First Seen"], "%Y-%m-%d %H:%M:%S")
                        
                        # Only keep today's records
                        if first_seen.date() == today:
                            temp_records[name] = {
                                "name": name,
                                "id": row["ID"],
                                "first_seen": first_seen,
                                "last_seen": datetime.strptime(row["Last Seen"], "%Y-%m-%d %H:%M:%S"),
                                "status": row["Status"],
                                "confidence": float(row["Confidence"]),
                                "method": row["Method"],
                                "entries": int(row["Entries"])
                            }
                        else:
                            old_count += 1
                    except:
                        continue
            
            self.records = temp_records
            
            # If we filtered out old records, update the files to show only today's data
            if old_count > 0:
                print(f"[INFO] Automatic Reset: Cleared {old_count} old records from previous days.")
                self.save_csv()
                self.save_excel()
                
        except Exception as e:
            print(f"[WARNING] Could not load existing CSV: {e}")

    def save_excel(self):
        """Save attendance records to Excel file with formatting."""
        try:
            data = []
            for record in self.records.values():
                data.append({
                    "Name": record["name"],
                    "ID": record["id"],
                    "First Seen": record["first_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Last Seen": record["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                    "Status": record["status"],
                    "Confidence": record["confidence"],
                    "Method": record["method"],
                    "Entries": record["entries"]
                })

            df = pd.DataFrame(data)

            with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Attendance', index=False)

                # Add summary sheet
                summary_data = {
                    "Metric": ["Total Present", "Date", "Report Generated"],
                    "Value": [
                        len(self.get_today_records()),
                        datetime.now().strftime("%Y-%m-%d"),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            return True, f"Saved to {self.excel_path}"
        except Exception as e:
            return False, f"Error saving Excel: {e}"

    def generate_daily_report(self):
        """Generate a formatted daily attendance report."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = self.get_today_records()

        report_lines = [
            "=" * 60,
            f"  DAILY ATTENDANCE REPORT - {today}",
            "=" * 60,
            f"  Total Present: {len(today_records)}",
            f"  Generated: {datetime.now().strftime('%H:%M:%S')}",
            "-" * 60,
            "  {:<20} {:<15} {:<20}".format("Name", "ID", "Time"),
            "-" * 60
        ]

        for name, record in today_records.items():
            time_str = record["first_seen"].strftime("%H:%M:%S")
            report_lines.append("  {:<20} {:<15} {:<20}".format(
                name, record["id"], time_str))

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = os.path.join(REPORTS_DIR, f"daily_report_{today}.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)

        return report_text, report_path

    def clear_daily_records(self):
        """Clear records for a new day."""
        self.records.clear()
        self.attendance_cooldown.clear()
        print("[INFO] Daily records cleared.")

    def _sync_to_master_log(self):
        """Ensure all current records exist in the master log (Initial Sync)."""
        if not os.path.exists(self.master_csv_path) and self.records:
            print("[INFO] Migrating existing records to master log...")
            for record in self.records.values():
                self._append_to_master_log(record)

    def _append_to_master_log(self, record):
        """Append a single record to the permanent master log."""
        file_exists = os.path.exists(self.master_csv_path)
        try:
            with open(self.master_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Name", "ID", "First Seen", "Method"])
                
                writer.writerow([
                    record["first_seen"].strftime("%Y-%m-%d"),
                    record["name"],
                    record["id"],
                    record["first_seen"].strftime("%H:%M:%S"),
                    record["method"]
                ])
        except Exception as e:
            print(f"[ERROR] Master log append failed: {e}")

    def get_history_report(self, days=120):
        """Aggregate attendance counts from the master log for the past N days."""
        if not os.path.exists(self.master_csv_path):
            return []

        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        history = {}  # name -> count

        try:
            with open(self.master_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        row_date = datetime.strptime(row["Date"], "%Y-%m-%d").date()
                        if row_date >= cutoff_date:
                            name = row["Name"]
                            history[name] = history.get(name, 0) + 1
                    except:
                        continue
            
            # Convert to list of dicts for JSON
            report = [{"name": name, "count": count} for name, count in history.items()]
            # Sort by count descending
            report.sort(key=lambda x: x["count"], reverse=True)
            return report
        except Exception as e:
            print(f"[ERROR] History report generation failed: {e}")
            return []

    def get_statistics(self):
        """Return attendance statistics."""
        today_records = self.get_today_records()
        return {
            "total_today": len(today_records),
            "total_all_time": len(self.records),
            "last_save": self.last_save.strftime("%H:%M:%S")
        }
