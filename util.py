import torch

def chamfer_distance(x, y):
    """
    Implementação nativa em PyTorch da Chamfer Distance para substituir o PyTorch3D.
    Substitui perfeitamente sem precisar instalar nada extra.
    """
    x_dist = torch.sum(x**2, dim=-1, keepdim=True) # (B, N, 1)
    y_dist = torch.sum(y**2, dim=-1, keepdim=True) # (B, M, 1)
    
    # Matriz de distância pareada (B, N, M)
    dist_matrix = x_dist + y_dist.transpose(-2, -1) - 2 * torch.bmm(x, y.transpose(-2, -1))
    dist_matrix = torch.clamp(dist_matrix, min=0.0)
    
    # Encontra os vizinhos mais próximos em ambas as direções
    loss_x, _ = torch.min(dist_matrix, dim=-1) # (B, N)
    loss_y, _ = torch.min(dist_matrix, dim=-2) # (B, M)
    
    loss = loss_x.mean(dim=-1) + loss_y.mean(dim=-1)
    
    # Retorna o loss médio e um placeholder None para manter a mesma assinatura do PyTorch3D
    return loss.mean()