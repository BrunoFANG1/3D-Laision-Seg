import torch
import pdb

def tensor_timeselection_and_expansion(tensor, t,  target_shape):
    # Convert 1-dim tensor to the shape of target shape (e.g. [B,C,H,W])
    tensor = tensor[t] # Only select target timesteps
    for i in range(len(target_shape)):
        if len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(0)
    expanded_tensor = tensor.expand(target_shape)
    print(expanded_tensor)
    return expanded_tensor

class Diffusion_Model:
    def __init__(
            self,
            max_time_steps: int = 1000,
            betas = None,
            pred_model = None,
            ):
        
        self.max_time_steps = max_time_steps
        self.alphas = 1 - betas  
        self.alpha_hat = torch.cumprod(self.alphas, dim=-1)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat)
        self.model = pred_model # as noise prediction
        self.loss = torch.nn.MSELoss()

    def time_sampler(self, num_time_steps):
        times = torch.randint(0, self.max_time_steps, (num_time_steps,))
        return times

    def forward_process(self, x_0, noise, t):
        # q(x_t | x_0)
        alpha_hat = tensor_timeselection_and_expansion(self.alpha_hat, t, noise.shape)
        sqrt_one_minus_alpha_hat = tensor_timeselection_and_expansion(self.sqrt_one_minus_alpha_hat, t, noise.shape)
        x_t = alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
        return x_t

    def reverse_process(self, x_t, t):
        # prediction on epsilon(x_t, t)
        pred_noise = torch.rand_like(x_t) # Change the model
        return pred_noise
    
    def train(self, sample, t):
        
        # forward
        noise = torch.rand_like(sample)
        x_t = self.forward_process(sample, noise, t)
        pdb.set_trace()
        # reverse
        pred_noise = self.reverse_process(x_t, t)
        pdb.set_trace()
        cal_loss = self.loss(pred_noise, noise)
        print(cal_loss)
        return None
        


if __name__ =='__main__':
    test = Diffusion_Model(max_time_steps=10,
                           betas = torch.randint(0, 100, (10,))/100,
                           pred_model= None)
    sample = torch.rand(2, 1, 2, 2)
    num_times = 2
    test.train(sample=sample, t=2)