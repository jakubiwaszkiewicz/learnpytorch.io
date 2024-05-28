import torch
# Create a random tensor with shape (7, 7)
print("Exercise 1")
print("Create a random tensor with shape (7, 7)")
print()

tensor = torch.rand(size=(7, 7))

print("Answer:")
print(tensor)
print()
print("-------------------")
print("Exercise 2")
print("Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)")
print()

tensor1 = torch.rand(size=(1, 7))
tensor2 = torch.rand(size=(7, 7))
tensor3 = tensor1 @ tensor2

print("Answer:")
print(tensor3)
print(tensor3.shape)
print()
print("-------------------")
print("Exercise 3")
print("Create two random tensors of shape (2, 3) and send them both to the GPU. Set torch.manual_seed(1234) when creating the tensors.")

torch.manual_seed(1234)

ex3_tensor1 = torch.rand(size=(2,3))
ex3_tensor2 = torch.rand(size=(2,3))

print("Answer:")

print(ex3_tensor1)
print(ex3_tensor2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

ex3_tensor1 = ex3_tensor1.to(device)
ex3_tensor2 = ex3_tensor2.to(device)

print(ex3_tensor1)
print(ex3_tensor2)

print()
print("-------------------")
print("Exercise 4")
print("Perform a matrix multiplication on the tensors you created in 3.")
print()
ex4_tensor1 = ex3_tensor1
ex4_tensor2 = ex3_tensor2
ex4_tensor2 = torch.reshape(ex4_tensor2, (3,2))
print(ex4_tensor1)
print(ex4_tensor2)
print("Answer:")
ex4_answer = ex4_tensor1 @ ex4_tensor2
print(ex4_tensor1 @ ex4_tensor2)
print()
print("-------------------")
print("Exercise 5")
print("Find the maximum and minimum values of the output of 4.")
print()
ex5_tensor = ex4_answer
print(ex5_tensor)
print("Answer:")
print("Max:")
print(ex5_tensor.max().item())
print("Min:")
print(ex5_tensor.min().item())
print()
print("-------------------")
print("Exercise 6")
print("Find the maximum and minimum index values of the output of 4.")
print()
ex6_tensor = ex5_tensor
print(ex6_tensor)
print("Answer:")
print()
print("Max:")
print(ex6_tensor.argmax())
print("Min:")
print(ex6_tensor.argmin())
print()
print("-------------------")
print("Exercise 7")
print("Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.")
print()
ex7_tensor = torch.rand(size=(1,1,1,10))
ex7_tensor_reshaped = torch.reshape(ex7_tensor, (10,))
print("Answer:")
print(ex7_tensor)
print(ex7_tensor_reshaped)
print()

# Also done this two extra-curriculum from website

# Spend 1-hour going through the PyTorch basics tutorial (I'd recommend the Quickstart and Tensors sections).
# To learn more on how a tensor can represent data, see this video: What's a tensor?