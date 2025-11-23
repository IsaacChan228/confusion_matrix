# User's active file:

"""
Read a confusion-matrix CSV and save a confusion-matrix image (heatmap).

Usage examples:
  python3 confusion_matrix.py -i "confusion matrix.csv"
  python3 confusion_matrix.py -i "confusion matrix.csv" -o out.png --no-annot --cmap plasma

The script will try to auto-detect whether the CSV has row labels (index) or not.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _size_to_float(val, default=10.0):
	"""Convert a rcParams size (which may be numeric or a name) to a float in points.

	Accepts numbers, numeric strings, or common size names used by matplotlib.
	"""
	# handle sequences
	try:
		import numpy as _np
		if isinstance(val, (_np.ndarray, list, tuple)):
			if len(val) > 0:
				val = val[0]
	except Exception:
		pass

	if isinstance(val, (int, float)):
		return float(val)
	if val is None:
		return float(default)
	s = str(val).strip()
	# try numeric string
	try:
		return float(s)
	except Exception:
		pass

	# map common named sizes to approximate point sizes
	mapping = {
		'xx-small': 6.0,
		'x-small': 8.0,
		'small': 10.0,
		'medium': 12.0,
		'large': 14.0,
		'x-large': 16.0,
		'xx-large': 18.0,
	}
	return float(mapping.get(s.lower(), default))


def _label_sort_key(x: str):
	s = str(x)
	# mark background to place it last
	is_bg = (s.lower() == 'background')
	# try integer
	try:
		n = int(s)
		return (is_bg, 0, n, '')
	except Exception:
		pass
	# try float
	try:
		f = float(s)
		return (is_bg, 0, f, '')
	except Exception:
		pass
	# fallback: non-numeric, sort lexicographically (case-insensitive)
	return (is_bg, 1, 0, s.lower())


def load_confusion_csv(path: str) -> pd.DataFrame:
	"""Try several reasonable ways to load the CSV into a square DataFrame.

	The function will attempt:
	1. read_csv(index_col=0) and check if square
	2. read_csv(header=None) and check if square
	3. detect long-form table with columns like 'Ground truth', 'Predicted' and 'Quantity'
	   and pivot it into a square confusion matrix
	4. read_csv() and if first column looks like labels (cols == rows+1), use it

	Raises ValueError if a square numeric matrix cannot be found.
	"""
	# 1) try index_col=0
	try:
		df = pd.read_csv(path, index_col=0)
		if df.shape[0] == df.shape[1]:
			return df
	except Exception:
		df = None

	# 2) try header=None (pure numeric grid)
	try:
		df2 = pd.read_csv(path, header=None)
		if df2.shape[0] == df2.shape[1]:
			# give default labels
			n = df2.shape[0]
			labels = [str(i) for i in range(n)]
			df2.columns = labels
			df2.index = labels
			return df2
	except Exception:
		df2 = None

	# 3) try reading without index and look for long-form (actual/predicted/count)
	try:
		df3 = pd.read_csv(path)
		# Normalize column names for detection
		cols_norm = [str(c).lower().strip() for c in df3.columns]

		# find candidate columns
		def find_col(preds):
			for p in preds:
				for i, c in enumerate(cols_norm):
					if p in c:
						return i
			return None

		actual_i = find_col(['ground', 'actual', 'ground truth', 'truth'])
		pred_i = find_col(['predict', 'predicted', 'prediction'])
		qty_i = find_col(['quant', 'count', 'quantity', 'value'])

		if actual_i is not None and pred_i is not None and qty_i is not None:
			actual_col = df3.columns[actual_i]
			pred_col = df3.columns[pred_i]
			qty_col = df3.columns[qty_i]
			pivot = pd.pivot_table(df3, index=actual_col, columns=pred_col, values=qty_col, aggfunc='sum', fill_value=0)
			# ensure consistent str labels for index/columns
			pivot.index = pivot.index.map(str)
			pivot.columns = pivot.columns.map(str)
			# reindex to full union of labels to keep square
			labels = sorted(set(pivot.index).union(set(pivot.columns)), key=_label_sort_key)
			pivot = pivot.reindex(index=labels, columns=labels, fill_value=0)
			return pivot

		# 4) if cols == rows + 1, assume first column is row labels
		if df3.shape[1] == df3.shape[0] + 1:
			labels = df3.iloc[:, 0].astype(str)
			mat = df3.iloc[:, 1:]
			mat.index = labels
			return mat

		# if already square
		if df3.shape[0] == df3.shape[1]:
			return df3
	except Exception:
		pass

	raise ValueError(f"Could not parse a square confusion matrix from '{path}'")


def plot_and_save(df: pd.DataFrame, out_path: str, title: Optional[str] = None, cmap: str = "Blues",
				  annot: bool = True, fmt: str = ".0f", dpi: int = 300, normalized: bool = False) -> None:
	# transpose so that columns (x-axis) are Ground truth and rows (y-axis) are Predicted
	values = df.values.astype(float).T
	nrows, ncols = values.shape

	fig, ax = plt.subplots(figsize=(max(4, ncols), max(4, nrows)))
	# ensure figure and axes background are white
	try:
		fig.patch.set_facecolor('white')
		ax.set_facecolor('white')
	except Exception:
		pass
	im = ax.imshow(values, cmap=cmap, aspect="equal")
	# make colorbar slimmer and slightly shorter so it doesn't dominate the figure
	cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, shrink=0.85)
	cbar_label = 'Percentage' if normalized else 'Count'
	cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=8)

	# determine font sizes and scale when normalized
	base_xtick = _size_to_float(plt.rcParams.get('xtick.labelsize', 10))
	base_ytick = _size_to_float(plt.rcParams.get('ytick.labelsize', base_xtick))
	base_axis = _size_to_float(plt.rcParams.get('axes.labelsize', 12))
	base_title = _size_to_float(plt.rcParams.get('axes.titlesize', 14))
	base_annot = _size_to_float(plt.rcParams.get('font.size', 10))
	base_cbar_label = _size_to_float(plt.rcParams.get('axes.labelsize', 12))
	base_cbar_ticks = _size_to_float(plt.rcParams.get('xtick.labelsize', 10))

	# apply the same font-size scaling to both normalized and non-normalized versions
	mult = 2.0

	# set colorbar font sizes
	try:
		cbar.ax.yaxis.label.set_size(int(base_cbar_label * mult))
		cbar.ax.tick_params(labelsize=int(base_cbar_ticks * mult * 0.8))
	except Exception:
		pass

	ax.set_xticks(np.arange(ncols))
	ax.set_yticks(np.arange(nrows))
	# after transpose: columns correspond to original index (Ground truth), rows to original columns (Predicted)
	ax.set_xticklabels(df.index, rotation=45, ha='right', fontsize=int(base_xtick * mult))
	ax.set_yticklabels(df.columns, fontsize=int(base_ytick * mult))

	ax.set_xlabel('Ground truth')
	ax.set_ylabel('Predicted')
	try:
		ax.xaxis.label.set_size(int(base_axis * mult))
		ax.yaxis.label.set_size(int(base_axis * mult))
	except Exception:
		pass
	if title:
		try:
			ax.set_title(title, fontsize=int(base_title * mult))
		except Exception:
			ax.set_title(title)

	if annot:
		vmax = np.nanmax(values)
		for i in range(nrows):
			for j in range(ncols):
				val = values[i, j]
				# format the value first, then hide if the formatted result represents zero
				try:
					text = f"{val:{fmt}}"
				except Exception:
					text = str(val)
				# clean text for numeric check
				_text_clean = str(text).strip().replace(',', '')
				_is_zero = False
				try:
					# if formatted text converts to float zero, treat as zero
					if _text_clean == '':
						_is_zero = True
					else:
						_num = float(_text_clean)
						_is_zero = math.isclose(_num, 0.0, abs_tol=1e-9)
				except Exception:
					# fallback: if original value is extremely close to zero
					_is_zero = math.isclose(val, 0.0, abs_tol=1e-9)
				if _is_zero:
					text = ''
					# draw a white rectangle over the cell to hide colormap for zero cells
					try:
						rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', edgecolor='none', zorder=2)
						ax.add_patch(rect)
					except Exception:
						pass
				color = 'white' if (not math.isnan(vmax) and val > vmax / 2.0) else 'black'
				ax.text(j, i, text, ha='center', va='center', color=color, fontsize=int(base_annot * mult), zorder=3)

	fig.tight_layout()
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	# save with white background
	fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description='Convert a confusion-matrix CSV to an image')
	parser.add_argument('-i', '--input', required=True, help='Path to CSV file (e.g. "confusion matrix.csv")')
	parser.add_argument('-o', '--output', default=None, help='Output image path (PNG). Defaults to input basename + .png')
	parser.add_argument('--cmap', default='Blues', help='Matplotlib colormap name (default: Blues)')
	parser.add_argument('--no-annot', dest='annot', action='store_false', help='Disable numeric annotations on cells')
	parser.add_argument('--fmt', default='.0f', help='Number format for annotations (default: .0f)')
	parser.add_argument('--title', default=None, help='Optional plot title')
	parser.add_argument('--dpi', type=int, default=300, help='Output image DPI (default: 300)')
	parser.add_argument('--normalize', action='store_true', help='Normalize by each ground-truth total (row-wise)')

	args = parser.parse_args()

	in_path = args.input
	if args.output:
		out_path = args.output
	else:
		base = os.path.splitext(in_path)[0].replace(' ', '_')
		out_path = base + ("_normalized.png" if args.normalize else ".png")

	try:
		df = load_confusion_csv(in_path)
	except Exception as e:
		print(f"Error reading CSV: {e}")
		raise

	# try to coerce values to numeric where possible
	try:
		df = df.apply(pd.to_numeric)
	except Exception:
		pass

	if args.normalize:
		# normalize by ground-truth totals (each row is a ground truth)
		# avoid division by zero by replacing zero sums with 1
		row_sums = df.sum(axis=1)
		row_sums_safe = row_sums.replace(0, 1)
		df = df.div(row_sums_safe, axis=0)
		# coerce all values to numeric (turn non-convertible -> NaN) then fill with 0
		df = df.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
		# convert to percentages (e.g. 0.12 -> 12) for display
		df = df * 100.0
		# if user didn't explicitly set annotation format, show integer percentages
		if args.fmt == '.0f':
			args.fmt = '.0f'

	plot_and_save(df, out_path, title=args.title, cmap=args.cmap, annot=args.annot, fmt=args.fmt, dpi=args.dpi, normalized=args.normalize)
	print(f"Saved confusion matrix image to: {out_path}")


if __name__ == '__main__':
	main()

