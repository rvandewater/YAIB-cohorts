# Conda Environment Setup Notes

To successfully run the conda environment, you may need to address the following setup steps:

---

## 🔧 Update Required Libraries

It may be necessary to update `libstdcxx-ng` to avoid compatibility issues:

```bash
conda update -c conda-forge libstdcxx-ng
```

---

## ⚠️ Common Issue: `units` Package

It is common to encounter errors related to the `units` package. Here's how to resolve them:

### ✅ Install the `units` Package

```bash
conda install -c conda-forge r-units
```

### 📦 System Dependency: `udunits2`

The `units` package depends on the `udunits2` C library. If it's missing, install it using:

```bash
conda install -c conda-forge udunits2
```

---

## 📁 `renv` Integration

Since you're using `renv` for package management in R, you might need to explicitly add the `units` package to your project:

```r
renv::install("units")
```
