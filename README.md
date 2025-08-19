## ContextDyG: A Dynamic Graph Learning Method with Contextual Interaction Frequency Encoding

### Abstract
Graph learning has broad applications in fields such as social media analysis and recommendation systems. Static graph models struggle to effectively capture the dynamic characteristics of evolving structural relationships over time. As a result, developing dynamic graph learning methods capable of modeling temporal dependencies and structural changes has become a key focus.

However, it is difficult for existing dynamic graph learning methods to precisely capture the complex dynamics of interactions within node neighborhoods and fail to fully reveal the deep semantics and evolutionary patterns of neighborhood structures. To address these challenges, we propose a novel method ContextDyG to better representing dynamic graph by learning contextual information. It introduces an innovative contextual interaction frequency encoding module CIF and a feature attention fusion module FAF. Specifically, CIF constructs a multi-dimensional raw feature vector to characterize the contextual importance of each neighboring node, and these features are abstracted in the frequency domain. FAF leverages a time-aware multi-head attention mechanism to adaptively fuse the features produced by the encoder, enhancing the quality of representations for downstream tasks. We conduct experiments on six real-world datasets and compare ContextDyG with previous state-of-the-art (SOTA) models. The experiments show that ContextDyG outperforms all baseline models with an average rank improvement of 34%, achieving a new SOTA.
### Overview
<img width="1474" height="866" alt="图片1" src="https://github.com/user-attachments/assets/88e8d4f0-8b6d-4b49-bd22-58427f8b2807" />

### CIF Key Code
```
class NeighborCooccurrenceEncoder(nn.Module):
    def __init__(self, neighbor_co_occurrence_feat_dim: int, max_input_sequence_length: int, device: str = 'cpu', dropout_rate: float = 0.1):
        """
        Contextual Interaction Frequency (CIF) Encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of CIF features (encodings)
        :param max_input_sequence_length: int, maximal length of the input sequence for FFT.
        :param device: str, device
        :param dropout_rate: float, dropout rate
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.cif_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device
        self.dropout_rate = dropout_rate
        self.internal_max_seq_length = max_input_sequence_length

        # MLP_cif: 4 (raw CIF) -> hidden_dim -> cif_feat_dim
        hidden_dim_mlp = self.cif_feat_dim * 2 # Example hidden dimension
        self.mlp_cif = nn.Sequential(
            nn.Linear(4, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim_mlp, self.cif_feat_dim)
        )

        # Learnable temporal decay rate
        self.learnable_gamma = nn.Parameter(torch.tensor(0.1)) # Initialize with a small positive value

        # Frequency domain filter parameters
        # The feature dimension for freq_filter should match cif_feat_dim
        self.freq_filter = nn.Parameter(
            torch.randn(1, self.internal_max_seq_length // 2 + 1, self.cif_feat_dim, 2, dtype=torch.float32)
        )
        self.fft_dropout = nn.Dropout(self.dropout_rate)


    def _apply_freq_block(self, features_sequence: torch.Tensor) -> torch.Tensor:
        """
        Applies frequency domain enhancement to a sequence of features.
        :param features_sequence: Tensor, shape (batch_size, seq_len, feat_dim)
        :return: Tensor, shape (batch_size, seq_len, feat_dim)
        """
        batch_size, seq_len, feat_dim = features_sequence.shape

        # Pad sequence to internal_max_seq_length if necessary for FFT, or use seq_len if dynamic FFT length is preferred
        # For now, assume internal_max_seq_length is appropriate and features_sequence might be shorter.
        # If seq_len < self.internal_max_seq_length, padding might be needed before FFT or use n=seq_len in rfft.
        # Let's use n=self.internal_max_seq_length for fixed-size FFT processing as in FilterLayer.
        
        x_fft = torch.fft.rfft(features_sequence, n=self.internal_max_seq_length, dim=1, norm='forward')
        
        # Ensure filter matches dimensions. If feat_dim is not self.cif_feat_dim, this is an issue.
        # Assuming feat_dim == self.cif_feat_dim as per mlp_cif output.
        filter_weights = torch.view_as_complex(self.freq_filter) # Shape (1, internal_max_seq_length//2+1, cif_feat_dim)
        
        # Adjust filter if its feature dimension doesn't match x_fft's feature dimension
        if filter_weights.shape[2] != x_fft.shape[2]:
             # This case should ideally not happen if mlp_cif outputs cif_feat_dim
             # For robustness, one might add a projection or ensure dimensions match.
             # For now, we assume they match based on design.
             pass

        filtered_x_fft = x_fft * filter_weights
        
        sequence_emb_ifft = torch.fft.irfft(filtered_x_fft, n=self.internal_max_seq_length, dim=1, norm='forward')
        
        # Truncate back to original sequence length
        sequence_emb_ifft_truncated = sequence_emb_ifft[:, :seq_len, :]
        
        hidden_states = self.fft_dropout(sequence_emb_ifft_truncated)
        # Residual connection
        enhanced_features = hidden_states + features_sequence
        
        return enhanced_features

    def forward(self, 
                src_padded_nodes_neighbor_ids: np.ndarray, 
                dst_padded_nodes_neighbor_ids: np.ndarray,
                src_padded_nodes_neighbor_times: np.ndarray,
                dst_padded_nodes_neighbor_times: np.ndarray,
                src_ego_ids_batch: np.ndarray,
                dst_ego_ids_batch: np.ndarray,
                current_interaction_times_batch: np.ndarray
                ) -> tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = src_padded_nodes_neighbor_ids.shape[0]
        src_seq_len = src_padded_nodes_neighbor_ids.shape[1]
        dst_seq_len = dst_padded_nodes_neighbor_ids.shape[1]

        src_raw_cif_vectors_batch = []
        dst_raw_cif_vectors_batch = []

        for i in range(batch_size):
            current_src_ego = src_ego_ids_batch[i]
            current_dst_ego = dst_ego_ids_batch[i]
            current_t = current_interaction_times_batch[i]

            # --- Process Source Ego's Neighbors ---
            src_neighbor_ids_sample = src_padded_nodes_neighbor_ids[i]
            src_neighbor_times_sample = src_padded_nodes_neighbor_times[i]
            # For C_L and C_X, consider only non-padded actual neighbors for counting
            actual_src_neighbors = src_neighbor_ids_sample[src_neighbor_ids_sample != 0]
            actual_dst_neighbors = dst_padded_nodes_neighbor_ids[i][dst_padded_nodes_neighbor_ids[i] != 0]
            
            # Pre-calculate counts for actual neighbors to optimize inner loop
            # For C_L: count of n_sp in s_ego's history
            src_neighbor_counts_in_src_history = {val: count for val, count in zip(*np.unique(actual_src_neighbors, return_counts=True))}
            # For C_X: count of n_sp in d_ego's history
            src_neighbor_counts_in_dst_history = {val: count for val, count in zip(*np.unique(actual_dst_neighbors, return_counts=True))}

            sample_src_raw_cif = []
            for k_s in range(src_seq_len):
                n_sp = src_neighbor_ids_sample[k_s]
                t_sp = src_neighbor_times_sample[k_s]

                if n_sp == 0: # Padded neighbor
                    sample_src_raw_cif.append([0.0, 0.0, 0.0, 0.0])
                    continue

                c_l = float(src_neighbor_counts_in_src_history.get(n_sp, 0))
                c_x = float(src_neighbor_counts_in_dst_history.get(n_sp, 0))
                m_val = 1.0 if n_sp == current_dst_ego else 0.0
                
                temporal_decay = 0.0
                if t_sp > 0 and current_t > t_sp: # Ensure valid time difference
                    # Using .abs() for learnable_gamma as it's a rate
                    temporal_decay = torch.exp(-torch.abs(self.learnable_gamma) * (current_t - t_sp)).item() 
                
                sample_src_raw_cif.append([c_l, c_x, m_val, temporal_decay])
            src_raw_cif_vectors_batch.append(sample_src_raw_cif)

            # --- Process Destination Ego's Neighbors (Symmetric) ---
            dst_neighbor_ids_sample = dst_padded_nodes_neighbor_ids[i]
            dst_neighbor_times_sample = dst_padded_nodes_neighbor_times[i]
            # For C_L and C_X for destination's perspective
            # C_L (dst): count of n_dp in d_ego's history
            dst_neighbor_counts_in_dst_history = {val: count for val, count in zip(*np.unique(actual_dst_neighbors, return_counts=True))}
            # C_X (dst): count of n_dp in s_ego's history
            dst_neighbor_counts_in_src_history = {val: count for val, count in zip(*np.unique(actual_src_neighbors, return_counts=True))}

            sample_dst_raw_cif = []
            for k_d in range(dst_seq_len):
                n_dp = dst_neighbor_ids_sample[k_d]
                t_dp = dst_neighbor_times_sample[k_d]

                if n_dp == 0: # Padded neighbor
                    sample_dst_raw_cif.append([0.0, 0.0, 0.0, 0.0])
                    continue
                
                # C_L from d_ego's perspective: count of n_dp in d_ego's historical neighbors
                c_l_dst = float(dst_neighbor_counts_in_dst_history.get(n_dp, 0))
                # C_X from d_ego's perspective: count of n_dp in s_ego's historical neighbors
                c_x_dst = float(dst_neighbor_counts_in_src_history.get(n_dp, 0))
                m_val_dst = 1.0 if n_dp == current_src_ego else 0.0
                
                temporal_decay_dst = 0.0
                if t_dp > 0 and current_t > t_dp: # Ensure valid time difference
                    temporal_decay_dst = torch.exp(-torch.abs(self.learnable_gamma) * (current_t - t_dp)).item()

                sample_dst_raw_cif.append([c_l_dst, c_x_dst, m_val_dst, temporal_decay_dst])
            dst_raw_cif_vectors_batch.append(sample_dst_raw_cif)

        # Convert lists of lists to tensors
        src_raw_cif_tensor = torch.tensor(src_raw_cif_vectors_batch, dtype=torch.float32, device=self.device)
        dst_raw_cif_tensor = torch.tensor(dst_raw_cif_vectors_batch, dtype=torch.float32, device=self.device)
        
        # Pass through MLP
        src_mlp_out = self.mlp_cif(src_raw_cif_tensor)
        dst_mlp_out = self.mlp_cif(dst_raw_cif_tensor)

        # Apply frequency block
        src_final_cif = self._apply_freq_block(src_mlp_out)
        dst_final_cif = self._apply_freq_block(dst_mlp_out)
        
        # Ensure padded positions remain zero (redundant if raw CIF for padding is all zeros and MLP preserves zeros)
        # However, for safety, explicitly zero them out based on original padding.
        # Create boolean masks for padded elements
        src_padding_mask = torch.from_numpy(src_padded_nodes_neighbor_ids == 0).to(self.device)
        dst_padding_mask = torch.from_numpy(dst_padded_nodes_neighbor_ids == 0).to(self.device)

        # Expand mask dimensions to match feature dimensions for broadcasting
        src_final_cif[src_padding_mask] = 0.0
        dst_final_cif[dst_padding_mask] = 0.0
        
        return src_final_cif, dst_final_cif

```
### Usage
The ContextDyG provided by this warehouse contains three variants, ContextDyG, w/o CIF, and w/o FAF. When using it, the corresponding variant needs to be renamed to ContextDyG.
