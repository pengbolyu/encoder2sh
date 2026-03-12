import argparse
import csv
import re
import statistics as stats
from pathlib import Path


def parse_fold_summary(log_text):
	# Supports both formats:
	# 1) Val Loss: ...
	# 2) Val Loss(best): ... , Val LSD Raw(best): ...
	pattern = re.compile(
		r"Fold\s+(\d+)\s+Summary\s*->\s*"
		r"Train Loss:\s*([0-9.]+),\s*"
		r"Val Loss(?:\(best\))?:\s*([0-9.]+),\s*"
		r"(?:Val LSD Raw(?:\(best\))?:\s*([0-9.]+),\s*)?"
		r"Test Loss:\s*([0-9.]+),\s*"
		r"Test LSD Smooth:\s*([0-9.]+),\s*"
		r"Test LSD Raw:\s*([0-9.]+)"
	)

	rows = []
	for match in pattern.finditer(log_text):
		fold = int(match.group(1))
		train_loss = float(match.group(2))
		val_loss = float(match.group(3))
		val_lsd_raw_best = float(match.group(4)) if match.group(4) is not None else float("nan")
		test_loss = float(match.group(5))
		test_lsd_smooth = float(match.group(6))
		test_lsd_raw = float(match.group(7))

		rows.append(
			{
				"fold": fold,
				"train_loss": train_loss,
				"val_loss_best": val_loss,
				"val_lsd_raw_best": val_lsd_raw_best,
				"test_loss": test_loss,
				"test_lsd_smooth": test_lsd_smooth,
				"test_lsd_raw": test_lsd_raw,
			}
		)

	rows.sort(key=lambda item: item["fold"])
	return rows


def summarize(rows, key):
	values = [row[key] for row in rows]
	mean_val = stats.mean(values)
	std_val = stats.pstdev(values) if len(values) > 1 else 0.0
	best = min(rows, key=lambda row: row[key])
	worst = max(rows, key=lambda row: row[key])
	return {
		"metric": key,
		"mean": mean_val,
		"std": std_val,
		"best_fold": best["fold"],
		"best_value": best[key],
		"worst_fold": worst["fold"],
		"worst_value": worst[key],
	}


def print_table(rows):
	header = (
		f"{'Fold':>4} | {'TrainLoss':>10} | {'ValLoss(best)':>12} | "
		f"{'ValLSDRaw(best)':>15} | {'TestLoss':>9} | {'TestLSDSmooth':>13} | {'TestLSDRaw':>10}"
	)
	print(header)
	print("-" * len(header))
	for row in rows:
		print(
			f"{row['fold']:>4} | "
			f"{row['train_loss']:>10.4f} | "
			f"{row['val_loss_best']:>12.4f} | "
			f"{row['val_lsd_raw_best']:>15.4f} | "
			f"{row['test_loss']:>9.4f} | "
			f"{row['test_lsd_smooth']:>13.4f} | "
			f"{row['test_lsd_raw']:>10.4f}"
		)


def print_summary(rows):
	metrics = [
		"train_loss",
		"val_loss_best",
		"val_lsd_raw_best",
		"test_loss",
		"test_lsd_smooth",
		"test_lsd_raw",
	]
	print("\nOverall Statistics")
	for key in metrics:
		summary = summarize(rows, key)
		print(
			f"- {key}: mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
			f"best=Fold {summary['best_fold']} ({summary['best_value']:.4f}), "
			f"worst=Fold {summary['worst_fold']} ({summary['worst_value']:.4f})"
		)


def save_csv(rows, out_csv):
	fieldnames = [
		"fold",
		"train_loss",
		"val_loss_best",
		"val_lsd_raw_best",
		"test_loss",
		"test_lsd_smooth",
		"test_lsd_raw",
	]
	with open(out_csv, "w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", required=True, help="Path to training_log file")
	parser.add_argument("--out_csv", default="fold_metrics_summary.csv", help="CSV output path")
	args = parser.parse_args()

	log_path = Path(args.log)
	if not log_path.exists():
		raise FileNotFoundError(f"Log file not found: {log_path}")

	text = log_path.read_text(encoding="utf-8")
	rows = parse_fold_summary(text)
	if not rows:
		raise RuntimeError("No Fold Summary rows matched; please check log format")

	print_table(rows)
	print_summary(rows)
	save_csv(rows, args.out_csv)
	print(f"\nCSV saved to: {args.out_csv}")


if __name__ == "__main__":
	main()
