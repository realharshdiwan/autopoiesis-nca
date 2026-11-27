import torch
import torch.nn as nn
import torch.nn.functional as F


class NCA2D(nn.Module):
    """
    Stochastic Neural Cellular Automata (NCA) with 16 channels:
      - Channels 0:3: Visible RGBA (R, G, B, alpha)
      - Channels 4:15: Hidden metabolic/signaling channels (12 channels)

    Architecture (~8,320 trainable parameters):
      - Perception: Depthwise Conv2d(16, 48, kernel_size=3, padding=1, groups=16, bias=False)
                    Fixed Sobel + Identity kernels (non-trainable)
      - Update MLP: Conv2d(48, 128, kernel_size=1) -> ReLU -> Conv2d(128, 16, kernel_size=1, bias=False)
                    Final layer initialized to exact zeros.
    """

    def __init__(self, channel_n: int = 16, hidden_n: int = 128, p_fire: float = 0.5):
        super().__init__()
        self.channel_n = channel_n
        self.hidden_n = hidden_n
        self.p_fire = p_fire

        # 1. Perception Layer (Fixed)
        self.perception = nn.Conv2d(
            in_channels=channel_n,
            out_channels=channel_n * 3,
            kernel_size=3,
            padding=1,
            groups=channel_n,
            bias=False,
        )
        self._init_perception_filters()

        # 2. Update MLP (Trainable: 48*128 + 128 + 128*16 = 6,272 + 2,048 = 8,320 parameters)
        self.fc1 = nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False)

        # Initialize final layer to exact zeros (identity behavior at t=0)
        nn.init.zeros_(self.fc2.weight)

    def _init_perception_filters(self):
        # Identity kernel K_I
        k_identity = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        )
        # Normalized Sobel filters
        k_sobel_x = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            / 8.0
        )
        k_sobel_y = k_sobel_x.t()

        weights = torch.zeros(
            (self.channel_n * 3, 1, 3, 3), dtype=torch.float32
        )
        for c in range(self.channel_n):
            weights[c * 3 + 0, 0] = k_identity
            weights[c * 3 + 1, 0] = k_sobel_x
            weights[c * 3 + 2, 0] = k_sobel_y

        self.perception.weight.data.copy_(weights)
        self.perception.weight.requires_grad = False

    def alive_mask(self, s: torch.Tensor) -> torch.Tensor:
        """
        Viability gating: cell is alive if max alpha in its 3x3 Moore neighborhood > 0.1
        """
        alpha = s[:, 3:4]
        return (F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1).float()

    def perceive(self, s: torch.Tensor) -> torch.Tensor:
        return self.perception(s)

    def forward(
        self,
        s: torch.Tensor,
        sever_mask: torch.Tensor = None,
        fire_mask: torch.Tensor = None,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """
        Single transition step:
          s_{t+1} = alive_t * (s_t + m_t * f_theta(z_t))

        Args:
            s: Current state tensor [B, 16, H, W]
            sever_mask: Optional boolean/float mask [1, 1, H, W] or [B, 1, H, W]
                        where hidden channels 4:16 are set to zero (Causal Severing Hook)
            fire_mask: Optional explicit Bernoulli firing mask [B, 1, H, W]
            generator: Optional torch.Generator for deterministic RNG sampling
        """
        B, C, H, W = s.shape

        # 1. Alive mask at current step
        alive_t = self.alive_mask(s)

        # 2. Local perception
        z = self.perceive(s)

        # 3. Update residual
        ds = self.fc2(self.relu(self.fc1(z)))

        # 4. Stochastic firing mask
        if fire_mask is None:
            if generator is not None:
                rand_tensor = torch.rand((B, 1, H, W), device=s.device, generator=generator)
            else:
                rand_tensor = torch.rand((B, 1, H, W), device=s.device)
            m_t = (rand_tensor < self.p_fire).float()
        else:
            m_t = fire_mask

        # 5. Gated state update
        s_candidate = s + m_t * ds
        s_next = alive_t * s_candidate

        # 6. Clamp hidden channels to [-5.0, 5.0] for recurrent stability
        hidden = torch.clamp(s_next[:, 4:16], -5.0, 5.0)
        s_next = torch.cat([s_next[:, :4], hidden], dim=1)

        # 7. Causal Interface Severing Hook (M4)
        if sever_mask is not None:
            keep_mask = (~sever_mask.bool()).float()
            hidden_severed = s_next[:, 4:16] * keep_mask
            s_next = torch.cat([s_next[:, :4], hidden_severed], dim=1)

        return s_next

    @staticmethod
    def create_seed(
        batch_size: int = 1,
        H: int = 64,
        W: int = 64,
        device: str = "cpu",
        seed_pos: tuple = (32, 32),
    ) -> torch.Tensor:
        """
        Initializes single active cell at seed_pos with alpha=1.0 and hidden channels 4:16 = 1.0.
        All other sites are zero.
        """
        state = torch.zeros((batch_size, 16, H, W), dtype=torch.float32, device=device)
        cy, cx = seed_pos
        state[:, 3, cy, cx] = 1.0  # alpha = 1.0
        state[:, 4:16, cy, cx] = 1.0  # hidden metabolic channels = 1.0
        return state
