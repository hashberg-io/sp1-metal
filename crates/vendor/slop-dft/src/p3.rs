// Modified from slop-dft 6.4.0 by the sp1-metal project; see CHANGES.md.
use std::convert::Infallible;

pub use p3_dft::*;
use slop_algebra::TwoAdicField;
use slop_alloc::CpuBackend;
use slop_matrix::{bitrev::BitReversableMatrix, Matrix};
use slop_tensor::Tensor;

use crate::{Dft, DftOrdering};

impl<F: TwoAdicField> Dft<F, CpuBackend> for Radix2DitParallel {
    type Error = Infallible;
    fn coset_dft_into(
        &self,
        src: &Tensor<F, CpuBackend>,
        dst: &mut Tensor<F, CpuBackend>,
        shift: F,
        log_blowup: usize,
        ordering: crate::DftOrdering,
        dim: usize,
    ) -> Result<(), Self::Error> {
        assert_eq!(src.sizes().len(), 2);
        assert_eq!(dst.sizes().len(), 2);
        assert_eq!(dim, 0, "Radix2DitParallel only supports DFT along the first dimension");

        // The GPU encode, where it computes this call exactly. In checking mode
        // it runs and is then compared against the CPU result below, which is
        // the only way the parity claim is made against real traces rather than
        // against test shapes.
        #[cfg(feature = "metal")]
        let checked = {
            let ran = crate::metal::coset_dft_into(src, dst, shift, log_blowup, ordering, dim);
            if ran && !crate::metal::checking() {
                return Ok(());
            }
            ran.then(|| dst.as_slice().to_vec())
        };

        let dst_matrix = std::mem::take(dst);
        let mut dst_matrix: slop_matrix::dense::RowMajorMatrix<F> = dst_matrix.try_into().unwrap();

        // Clear the destination matrix.
        dst_matrix.values.clear();
        // Copy the source matrix to
        dst_matrix.values.extend_from_slice(src.as_slice());
        // initialize the rest of the entries to zeros.
        dst_matrix.values.resize(src.total_len() << log_blowup, F::zero());
        // Resize the destination matrix to the correct dimensions

        let result = self.coset_dft_batch(dst_matrix, shift);

        let result_matrix = match ordering {
            DftOrdering::Normal => result.to_row_major_matrix(),
            DftOrdering::BitReversed => result.bit_reverse_rows().to_row_major_matrix(),
        };

        let mut result_tensor: Tensor<F, CpuBackend> = result_matrix.into();
        std::mem::swap(dst, &mut result_tensor);

        #[cfg(feature = "metal")]
        if let Some(from_gpu) = checked {
            assert_eq!(
                from_gpu.len(),
                dst.as_slice().len(),
                "metal encode produced a codeword of the wrong length"
            );
            let at = from_gpu.iter().zip(dst.as_slice()).position(|(a, b)| a != b);
            assert!(at.is_none(), "metal encode disagrees with the CPU at index {}", at.unwrap());
        }

        Ok(())
    }
}
