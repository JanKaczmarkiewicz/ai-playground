# Neuron network

Neuron network is a tool to compress data into approximated function.

Very simple example:

for given input data:
```
inputs      expected output
0.2         0.4
0.3         0.6
0.5         1.0
```
after training neuron network will produce following function:
```
y = x * 1.99996732345 + 0.000001232314
```

This function (model) has `0.0000123` accuracy for input data.

The data had this property that `input * 2 = expected_output`.

## linear algebra column view of the problem

in linear algebra we could write problem as:

```
inputs          expected output
1.0 2.3 5.0     0.4
4.0 3.3 6.1     0.6
6.0 2.3 2.4     1.0
```

Column view:
```
  [1.0]     [2.3]     [5.0]    [1]   [0.4]        
w1[0.4] + w2[3.3] + w3[6.1] + b[1] = [0.6]
  [6.0]     [2.3]     [2.4]    [1]   [1.0]
```

What w1, w2, w3, b will give us correct results? In this case it might be none because this is to few connections to model this data.

## training process

To find function that approximates input data we create function that has random weigths:

```
y = x * 0.1 + 0.2
```

As you can see it does not perform well.

for example given 
```
0.2
```
it will produce
```
y = 0.22
```
which is far from expected `0.4`. 

Error in this case is `0.4 - 0.22 = 0.18`
Abstracted version of error fn is

`E = (y - expected)`

To change that we can nudge params a bit so that model will be more adjusted:

```
y = x * (0.1 + 0.6) + (0.2 - 0.1)
y = 0.14 + 0.1 = 0.28
0.4 - 0.28 = 0.12
```

Great, the error went down. But the change in parameters is choosen arbitrary. It would be nice to choose those values automatically.

We want to change `w` and `b` in such a way that this will reduce the error. Partial derivetives will help us with that:

```
dE/dw (x*w + b - expected) = x
dE/db (x*w + b - expected) = 1
```

