# Tarner and Muskat Production Forecasting

Python implementation of the Tarner/Turner and Muskat methods for cumulative oil and gas production forecasting.

## Features

- Tarner/Turner material-balance workflow
- Muskat first-year comparison
- PVT data embedded in Python
- krg/kro interpolation from total liquid saturation
- MBE and relative-permeability gas-production line intersection
- Cumulative oil and gas production forecast
- Intersection plots
- Excel result export

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python examples/run_tarner_muskat.py
```

## Outputs

The script generates:

- Tarner/Turner pressure-step results
- First-year Muskat comparison
- Intersection plots
- Final oil and gas production plots
- Excel output file

Generated Excel and plot files are ignored by Git.

## Data

The PVT table used in the calculation is embedded directly in the Python script. The original Excel workbook is not required to run the code.

## License

This project is released under the MIT License.
