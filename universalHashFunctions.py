import random

# return a random hash function from a universal family
# p = large prime number 
# m = modulus of arithmetic

def create_random_hash_function(p=2**33-355, m=2**32-1):
    a = random.randint(1,p-1)
    b = random.randint(0, p-1)
    return lambda x: 1 + (((a * x + b) % p) % m)

# usage examples...
# h1 = create_random_hash_function()
# h2 = create_random_hash_function()
# print(h1(23))
# print(h2(23))
