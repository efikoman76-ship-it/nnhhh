//! Cache-friendly dense-matrix substrate with data-parallel kernels.
//!
//! `Matrix` is a row-major `rows x cols` f32 matrix. Hot paths (matmul,
//! softmax, normalisation, activations) are parallelised over rows with
//! rayon and written to be auto-vectorisable: tight inner loops, no
//! branching, contiguous access.

use crate::error::{AetherError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Row-major dense f32 matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Allocate a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Allocate a matrix filled with `value`.
    pub fn fill(rows: usize, cols: usize, value: f32) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Build from a flat buffer, checking its length.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Matrix> {
        if data.len() != rows * cols {
            return Err(AetherError::ShapeMismatch(format!(
                "from_vec needs {} values for {rows}x{cols}, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// Build from row vectors, checking rectangularity.
    pub fn from_rows(rows: Vec<Vec<f32>>) -> Result<Matrix> {
        if rows.is_empty() {
            return Err(AetherError::EmptyInput("from_rows got no rows".to_string()));
        }
        let cols = rows[0].len();
        if cols == 0 {
            return Err(AetherError::EmptyInput("from_rows got empty rows".to_string()));
        }
        for (i, r) in rows.iter().enumerate() {
            if r.len() != cols {
                return Err(AetherError::ShapeMismatch(format!(
                    "row 0 has {cols} cols but row {i} has {}",
                    r.len()
                )));
            }
        }
        let data: Vec<f32> = rows.into_iter().flatten().collect();
        Ok(Matrix {
            rows: data.len() / cols,
            cols,
            data,
        })
    }

    /// Standard-normal entries from a deterministic seed (Box–Muller).
    pub fn randn_seeded(seed: u64, rows: usize, cols: usize) -> Matrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let total = rows * cols;
        let mut data = Vec::with_capacity(total);
        while data.len() < total {
            // u1 stays away from 0 so ln() is finite.
            let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            data.push(radius * theta.cos());
            if data.len() < total {
                data.push(radius * theta.sin());
            }
        }
        Matrix { rows, cols, data }
    }

    /// Xavier-uniform initialisation from a deterministic seed.
    pub fn xavier_seeded(seed: u64, rows: usize, cols: usize) -> Matrix {
        let bound = (6.0 / (rows + cols) as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-bound..bound)).collect();
        Matrix { rows, cols, data }
    }

    /// Outer product `a Tensored b`, used by Hebbian updates.
    pub fn outer(a: &[f32], b: &[f32]) -> Matrix {
        let mut data = Vec::with_capacity(a.len() * b.len());
        for x in a {
            for y in b {
                data.push(x * y);
            }
        }
        Matrix {
            rows: a.len(),
            cols: b.len(),
            data,
        }
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Total element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Borrow a row slice.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Mutably borrow a row slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Read one element.
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    /// Write one element.
    pub fn set(&mut self, i: usize, j: usize, v: f32) {
        self.data[i * self.cols + j] = v;
    }

    /// Overwrite row `i` with `values`.
    pub fn set_row(&mut self, i: usize, values: &[f32]) -> Result<()> {
        if values.len() != self.cols {
            return Err(AetherError::ShapeMismatch(format!(
                "set_row needs {} values, got {}",
                self.cols,
                values.len()
            )));
        }
        self.row_mut(i).copy_from_slice(values);
        Ok(())
    }

    /// Flat view of the buffer.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Flat mutable view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consume into the flat buffer.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    fn check_same_shape(&self, other: &Matrix, op: &str) -> Result<()> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(AetherError::ShapeMismatch(format!(
                "{op}: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        Ok(())
    }

    /// Cache-friendly parallel matmul (`self @ other`), k-inner loop order.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(AetherError::ShapeMismatch(format!(
                "matmul: {}x{} @ {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let out_cols = other.cols;
        let mut out = vec![0.0f32; self.rows * out_cols];
        out.par_chunks_mut(out_cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_base = i * self.cols;
                for k in 0..self.cols {
                    let aik = self.data[a_base + k];
                    let b_base = k * out_cols;
                    for j in 0..out_cols {
                        out_row[j] += aik * other.data[b_base + j];
                    }
                }
            });
        Ok(Matrix {
            rows: self.rows,
            cols: out_cols,
            data: out,
        })
    }

    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: out,
        }
    }

    /// Elementwise add.
    pub fn add(&self, other: &Matrix) -> Result<Matrix> {
        self.check_same_shape(other, "add")?;
        let data: Vec<f32> = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// In-place elementwise add.
    pub fn add_inplace(&mut self, other: &Matrix) -> Result<()> {
        self.check_same_shape(other, "add_inplace")?;
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, b)| *a += *b);
        Ok(())
    }

    /// In-place `self += scale * other`.
    pub fn add_scaled_inplace(&mut self, other: &Matrix, scale: f32) -> Result<()> {
        self.check_same_shape(other, "add_scaled_inplace")?;
        self.data
            .par_iter_mut()
            .zip(other.data.par_iter())
            .for_each(|(a, b)| *a += scale * *b);
        Ok(())
    }

    /// Scale every element.
    pub fn scale(&self, s: f32) -> Matrix {
        let data: Vec<f32> = self.data.par_iter().map(|x| x * s).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Numerically stable row-wise softmax.
    pub fn softmax_rows(&self) -> Matrix {
        let mut out = self.clone();
        softmax_rows_inplace(&mut out);
        out
    }

    /// Row-wise softmax where `mask[i][j] == false` forbids position j.
    ///
    /// Rows with no allowed position fall back to a uniform distribution so
    /// no NaN can ever escape.
    pub fn masked_softmax_rows(&self, mask: &[Vec<bool>]) -> Result<Matrix> {
        if mask.len() != self.rows {
            return Err(AetherError::ShapeMismatch(format!(
                "mask has {} rows, matrix has {}",
                mask.len(),
                self.rows
            )));
        }
        let mut out = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            if mask[i].len() != self.cols {
                return Err(AetherError::ShapeMismatch(format!(
                    "mask row {i} has {} cols, matrix has {}",
                    mask[i].len(),
                    self.cols
                )));
            }
            let src = self.row(i);
            let dst = out.row_mut(i);
            let mut max = f32::NEG_INFINITY;
            for j in 0..self.cols {
                if mask[i][j] && src[j] > max {
                    max = src[j];
                }
            }
            if max == f32::NEG_INFINITY {
                for v in dst.iter_mut() {
                    *v = 1.0 / self.cols as f32;
                }
                continue;
            }
            let mut sum = 0.0f32;
            for j in 0..self.cols {
                let e = if mask[i][j] { (src[j] - max).exp() } else { 0.0 };
                dst[j] = e;
                sum += e;
            }
            let inv = 1.0 / (sum + 1e-12);
            for v in dst.iter_mut() {
                *v *= inv;
            }
        }
        Ok(out)
    }

    /// Row-wise layer normalisation with affine gain and bias.
    pub fn layer_norm(&self, gamma: &[f32], beta: &[f32], eps: f32) -> Result<Matrix> {
        if gamma.len() != self.cols || beta.len() != self.cols {
            return Err(AetherError::ShapeMismatch(format!(
                "layer_norm needs gamma/beta of len {}, got {}/{}",
                self.cols,
                gamma.len(),
                beta.len()
            )));
        }
        let mut out = Matrix::zeros(self.rows, self.cols);
        out.data
            .par_chunks_mut(self.cols)
            .enumerate()
            .for_each(|(i, dst)| {
                let src = self.row(i);
                let mean = src.iter().sum::<f32>() / self.cols as f32;
                let var = src.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>()
                    / self.cols as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for j in 0..self.cols {
                    dst[j] = (src[j] - mean) * inv * gamma[j] + beta[j];
                }
            });
        Ok(out)
    }

    /// Exact GELU activation.
    pub fn gelu(&self) -> Matrix {
        self.map(|x| {
            0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
        })
    }

    /// SiLU / Swish activation.
    pub fn silu(&self) -> Matrix {
        self.map(|x| x / (1.0 + (-x).exp()))
    }

    /// ReLU activation.
    pub fn relu(&self) -> Matrix {
        self.map(|x| x.max(0.0))
    }

    /// Elementwise map.
    pub fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Matrix {
        let data: Vec<f32> = self.data.par_iter().map(|x| f(*x)).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    /// Mean vector over rows (the "gist" of a sequence).
    pub fn mean_over_rows(&self) -> Vec<f32> {
        let mut acc = vec![0.0f32; self.cols];
        for i in 0..self.rows {
            for (a, x) in acc.iter_mut().zip(self.row(i).iter()) {
                *a += *x;
            }
        }
        let inv = 1.0 / self.rows.max(1) as f32;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        acc
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f32>() / self.data.len() as f32
    }
}

/// In-place stable softmax used internally to avoid reallocations.
fn softmax_rows_inplace(m: &mut Matrix) {
    m.data.par_chunks_mut(m.cols).for_each(|row| {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / (sum + 1e-12);
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

/// Logistic sigmoid.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let id = Matrix::identity(2);
        let c = a.matmul(&id).unwrap();
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_known_product() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let b = Matrix::from_rows(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_mismatch() {
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(4, 2);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![-100.0, 0.0, 100.0]]).unwrap();
        let s = m.softmax_rows();
        for i in 0..2 {
            let sum: f32 = s.row(i).iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {i} sums to {sum}");
            assert!(s.row(i).iter().all(|x| *x >= 0.0));
        }
    }

    #[test]
    fn masked_softmax_respects_mask_and_fallback() {
        let m = Matrix::from_rows(vec![vec![5.0, 1.0], vec![5.0, 1.0]]).unwrap();
        let s = m
            .masked_softmax_rows(&[vec![true, false], vec![false, false]])
            .unwrap();
        assert!((s.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((s.get(0, 1)).abs() < 1e-5);
        assert!((s.get(1, 0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let m = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let n = m.layer_norm(&gamma, &beta, 1e-5).unwrap();
        let mean: f32 = n.row(0).iter().sum::<f32>() / 4.0;
        let var: f32 = n.row(0).iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn randn_is_deterministic_and_sane() {
        let a = Matrix::randn_seeded(7, 64, 64);
        let b = Matrix::randn_seeded(7, 64, 64);
        assert_eq!(a.as_slice(), b.as_slice());
        let mean = a.mean_all().abs();
        assert!(mean < 0.1, "mean {mean}");
        assert!(a.as_slice().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn transpose_is_involution() {
        let a = Matrix::randn_seeded(3, 4, 5);
        let b = a.transpose().transpose();
        assert_eq!(a.as_slice(), b.as_slice());
    }
}
