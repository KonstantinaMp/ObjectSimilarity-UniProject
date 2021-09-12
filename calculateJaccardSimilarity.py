# calculate the Jaccard similarity of 2 sets

s1 = {2,5,10,12,23}
s2 = {1,4,5,12,23,32}
Js = (len(s1.intersection(s2)) / len(s1.union(s2)))
print("Jaccard similarity of s1,s2 = %.2f" %Js)
   