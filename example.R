require(MASS);
require(FNN)
require(igraph)
require(Matrix)
require(ggplot2)



# Make some sample data ---------------------------------------------------
##  Let's make some sample data to play with. This is the section you would 
##  replace when using it with your own data. We will make four balls in d
##  dimensions, each with N/4 points. Each one will be separated by 1 in each
##  dimension, so they are actually very separable, even though in any two
##  dimensions they look like they are overlapping.

set.seed(3)
N <- 4E3;
d <- 100; 
input_data <- rbind(mvrnorm(n = N/4, rep(1, d), diag(d)),
                    mvrnorm(n = N/4, rep(2, d), diag(d)),
                    mvrnorm(n = N/4, rep(3, d), diag(d)),
                    mvrnorm(n = N/4, rep(4, d), diag(d))
)
labs <- as.factor(c(rep(1,N/4),
          rep(2,N/4),
          rep(3,N/4),
          rep(4,N/4)
          ))

ggplot(data.frame(V1=input_data[,1],V2=input_data[,2],color=labs), 
       aes(x=V1,y=V2, color=color)) + geom_point() 

# Construct a graph ---------------------------------------------------

# Connect every point with the kNN nearest points
kNN <- 50;
number_of_evecs <- 10;
knnout <- get.knn(input_data,k=kNN)

# The bandwidth of the Gaussian kernel will be the distance to the kNNth point,
# so it will be adaptive.
sigma <- knnout$nn.dist[,kNN]

# Construct a sparse matrix using the distances
i <- rep(1:N,each=kNN) 
j <- t(knnout$nn.index)[1:(nrow(knnout$nn.index)*ncol(knnout$nn.index))]
aff <- exp(-(knnout$nn.dist)^2/(sigma^2))
val <- t(aff)[1:N]
A <-  sparseMatrix(i=i,j=j,x=val)
A <- (A+t(A))/2;

# Let's visualize the affinity matrix. Comment this out if you have a lot of
# points, it might take a while. You won't see anything in real life data,
# because it's not sorted, but in this example dataset, you should see four
# blocks.
image(A)

# Make the affinity matrix into a transition/Markov matrix such that the rows
# sum to 1
D <- rowSums(A);
A2 <- sweep(x = A,MARGIN =  1,STATS = D,FUN = '/')

# Compute the top number_of_evecs using a sparse eigensolver based on ARPACK's
# implementation of Lanczos method
func <- function(x, extra=NULL) { as.vector(A2 %*% x) } 
spec <- arpack(func,options=list(n=N, nev=number_of_evecs,ncv=50, which="LM"), sym=FALSE)
spec$vectors <- Re(spec$vectors)

# Visualize results! ---------------------------------------------------

# It's always good to look at the eigenvalues. In the example, you will have a
# huge spectral gap after the 4th eigenvalue, as the graph is almost
# disconnected, with four components
plot(abs(spec$values))

# Here is the diffusion map embedding of this dataset!
ggplot(data.frame(V1=spec$vectors[,2],V2=spec$vectors[,3],color=labs),  # Change the 2 and 3 to be the number of the eigenvectors you want to visualize
       aes(x=V1,y=V2, color=color)) + geom_point() 

# After the fourth eigenvector, you just get noise 
# ggplot(data.frame(V1=spec$vectors[,5],V2=spec$vectors[,6],color=labs), 
       # aes(x=V1,y=V2, color=color)) + geom_point() 
