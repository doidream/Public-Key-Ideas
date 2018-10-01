from numpy import array,matrix,array_equal,zeros,identity
from scipy import linalg
from math import sqrt

def simpleIntDH(a,b,x,n):
    ret = x
    for i in range(n):
        ret = a*ret+b
    return ret

def simpleMatrixDH(a,b,x,n):
    ret = x
    for i in range(n):
        ret = (a*ret)+b
    return ret

def simpleVectorDH(a,b,x,n):
    ret = x
    for i in range(n):
        ret = array(a.dot(ret)+b).squeeze()
    return ret

def IntDH(a,b,x,n):
    return x*(a**n)+ b*(((a**n) -1 )/(a-1))

def MatrixDH(a,b,x,n):
    dim = a.ndim
    return (a**n)*x + (((a**n) - identity(dim))*linalg.inv(a-identity(dim)))*b

def VectorDH(a,b,x,n):
    dim = a.ndim
    return (a**(n)).dot(x) + ((a**(n) - identity(dim))*linalg.inv(a-identity(dim))).dot(b)

def MatrixModDH(a,b,x,n,N):
    dim = a.ndim
    return ((modPow(a,n,N)*(x%N))%N + ((modPow(a,n,N) - identity(dim))%N * modMatInv((a-identity(dim))%N,N))*(b%N))%N

def VectorModDH(a,b,x,n,N):
    dim = a.ndim
    return (modPow(a,n,N).dot(x%N) + ((modPow(a,n,N) - identity(dim))%N * modMatInv((a-identity(dim))%N,N)).dot(b%N))%N

def IntModDH(a,b,x,n,N):
    return ((x*modPow(a,n,N)) %N + b%N * ((modPow(a,n,N) - 1)%N * modInv(a-1,N)%N))%N


def simpleRatDiffDH(a,b,c,d,x,n):
    ret = x
    for i in range(n):
        ret = (a*ret+b)/(c*ret+d)
    return ret

def simpleQuadOneDH(a,b,x,n):
    ret = x
    for i in range(n):
        ret = a*(ret*ret) + b*ret + (b*b - 2*b)/(4*a)
    return ret

def simpleRatDiffModDH(a,b,c,d,x,n,N):
    ret = x
    for i in range(n):
        ret = ((a*ret+b)%N * modInv((c*ret+d)%N,N))%N
    return ret

def simpleQuadOneModDH(a,b,x,n,N):
    ret = x
    c = ((b*b - 2*b)%N * modInv(4*a,N))%N
    for i in range(n):
        ret = (a*(ret*ret) + b*ret+ c)%N
        #ret = ((a*(ret*ret) + b*ret)%N + (((b*b - 2*b)%N) * modInv(4*a,N))%N)%N
    return ret

def RatDiffDH(a,b,c,d,x,n):
    u = (a + d + sqrt((a-d)**2 + 4*b*c))/2
    v = (a + d - sqrt((a-d)**2 + 4*b*c))/2
    z = ((c*x - a + u)*(u**(n-1)) - (c*x - a + v)*(v**(n-1)))/((c*x - a + u)*(u**(n)) - (c*x - a + v)*(v**(n)))
    return (a/c) + ((b*c - a*d)/c)*z

def QuadOneDH(a,b,x,n):
    u = (2*a*x + b)/2
    return (2*(u**(2**n)) - b)/(2*a)

def RatDiffModDH(a,b,c,d,x,n,N):
    s = modSqrt(((a-d)**2 + 4*b*c)%N,N)
    if s is None or len(s) == 0:
        raise(((a-d)**2 + 4*b*c)%N," not quadratic residue")
    u = ((a + d + s[0])%N * modInv(2,N))%N
    v = ((a + d + s[1])%N * modInv(2,N))%N
    z = (((c*x - a + u)%N * modPow(u,(n-1),N) - (c*x - a + v)%N * modPow(v,(n-1),N))%N * modInv(((c*x - a + u)%N * modPow(u,n,N) - (c*x - a + v)%N * modPow(v,n,N))%N,N))%N
    return ((a*modInv(c,N))%N + ((b*c - a*d)%N * modInv(c,N))*z)%N

def QuadOneModDH(a,b,x,n,N):
    u = ((2*a*x + b)%N * modInv(2,N))%N
    return ((2*modPow(u,2**n,N) + (N - b))%N * modInv(2*a,N))%N

def modPow(x,n,N):
    if n <= 1:
        return x%N
    elif n%2 == 0:
        return modPow((x*x)%N,n/2,N)
    elif n%2 == 1:
        return (x%N)*modPow(x%N,n-1,N)
    else:
        raise Exception('modpow does not handle input n')
    
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modInv(a, m):
    v = a%m
    if v == 0:
        return 0
    g, x, y = egcd(v, m)
    if g != 1:
        print(v)
        raise Exception('modular inverse does not exist')
    else:
        return x % m

def modMatInv(A,p):       # Finds the inverse of matrix A mod p
  n=len(A)
  A=matrix(A)
  adj=zeros(shape=(n,n))
  for i in range(0,n):
    for j in range(0,n):
      adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
  return (modInv(int(round(linalg.det(A))),p)*adj)%p

def minor(A,i,j):    # Return matrix A with the ith row and jth column deleted
  A=array(A)
  minor=zeros(shape=(len(A)-1,len(A)-1))
  p=0
  for s in range(0,len(minor)):
    if p==i:
      p=p+1
    q=0
    for t in range(0,len(minor)):
      if q==j:
        q=q+1
      minor[s][t]=A[p][q]
      q=q+1
    p=p+1
  return minor

def legendre_symbol(a, p):
    """
    Legendre symbol
    Define if a is a quadratic residue modulo odd prime
    http://en.wikipedia.org/wiki/Legendre_symbol
    """
    ls = pow(a, int((p - 1)/2), p)
    if ls == p - 1:
        return -1
    return ls

def modSqrt(a, p):
    """
    Square root modulo prime number
    Solve the equation
        x^2 = a mod p
    and return list of x solution
    http://en.wikipedia.org/wiki/Tonelli-Shanks_algorithm
    """
    a %= p

    # Simple case
    if a == 0:
        return [0,0]
    if p == 2:
        return [a,a]

    # Check solution existence on odd prime
    if legendre_symbol(a, p) != 1:
        return []

    # Simple case
    if p % 4 == 3:
        x = pow(a, int((p + 1)/4), p)
        return [x, p-x]

    # Factor p-1 on the form q * 2^s (with Q odd)
    q, s = p - 1, 0
    while q % 2 == 0:
        s += 1
        q //= 2

    # Select a z which is a quadratic non resudue modulo p
    z = 1
    while legendre_symbol(z, p) != -1:
        z += 1
    c = pow(z, q, p)

    # Search for a solution
    x = pow(a, int((q + 1)/2), p)
    t = pow(a, int(q), p)
    m = s
    while t != 1:
        # Find the lowest i such that t^(2^i) = 1
        i, e = 0, 2
        for i in range(1, m):
            if pow(t, e, p) == 1:
                break
            e *= 2

        # Update next value to iterate
        b = pow(c, 2**(m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i

    return [x, p-x]

#Depends on a
#Generic try, a = (N+e)/2, e in {0,1,-1} 
def IntDHPeriod(a,b,x,N):
    for i in range(1,N+2):
        if IntModDH(a,b,x%N,i,N) == x%N:
            return i

def allIntDHPeriod(N):
    for a in range(1,N):
        for b in range(N):
            for x in range(N):
                print(a,b,x,IntDHPeriod(a,b,x,N))

def betterIntDHPeriod(N):
    for a in range(1,N):
        print(a,(N-1)/2,(N+1)/2,IntDHPeriod(a,(N-1)/2,(N+1)/2,N))

#Period P must satisfy GCD(P,N) != 1, but N/P does need not be an integer
#Depends on A, specific determinants have higher chance specific orders
#D = [[0,b],[c,d]], det = -b*c, only specific det and d will work
#D = [[0,b],[-det/b,det/b]] may always have order greater than N
#Try det = 2 or 2**k, try d = 5, highest 2**K or (N+-1)/2
#AGA^(-1) is the same order as G
def VectorDHPeriod(a,b,x,N):
    for i in range(1,(N**2)+1):
        if (VectorModDH(a,b,x%N,i,N) == x%N).all():
            return i
def fastVectorDHPeriod(b,x,N):
    for a1 in range(1,N):
        for a2 in range(1,N):
            for a3 in range(1,N):
                for a4 in range(1,N):
                    a = matrix([[a1,a2],[a3,a4]])
                    print(a,b,x,VectorDHPeriod(a,b,x,N))

#Dependent on A, try det(a) = 2**k
#RGR^(-1) same order as G
def MatrixDHPeriod(a,b,x,N):
    for i in range(1,N**9):
        if (MatrixModDH(a,b,x%N,i,N) == x%N).all():
            return i
def RatDiffDHPeriod(a,b,c,d,x,N):
    for i in range(1,N+1):
        if RatDiffModDH(a,b,c,d,x%N,i,N) == x%N:
            return i

#look for a*d-b*c = N-1 or (N+-1)/2 or 2**k
def fastRatDiffDHPeriod(x,N):
    for a in range(1,N):
        for b in range(1,N):
            for c in range(1,N):
                for d in range(1,N):
                    if legendre_symbol(((a-d)**2 + 4*b*c)%N,N) == 1:
                        print(a,b,c,d,(a*a*d*d-b*b*c*c)%N,x,RatDiffDHPeriod(a,b,c,d,x,N))
