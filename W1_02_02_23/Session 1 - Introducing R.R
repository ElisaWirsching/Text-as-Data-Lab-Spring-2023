# TA: Elisa Wirsching
# Course: Text as Data
# Date: 02/02/2023
# Lab adapted from: Lucia Motolinia, Pedro L. Rodriguez, Kevin Munger, Patrick Chester and Leslie Huang.

# Before you start:
# Download the file "national_clinton_trump_6_20_2016.csv" from the repo

# TIPS:
# you should always (always!) annotate your code
# use version control (for GitHub, see https://ourcodingclub.github.io/tutorials/git/)
# DEBUGGING: rubber duck it (https://en.wikipedia.org/wiki/Rubber_duck_debugging)
# Google is your friend. Type your question and add "R" to the end of it.
# knitr is useful for problem sets that require showing your code
# for (bigger) projects: use R projects or a dependency manager (packrat) 


# ============================================================================= #
####                                SETTING UP                               ####
# ============================================================================= #

# 1.1 Clearing environment
rm(list = ls())

# 1.2 Working directory

# you can get and set the working directory manually (this is pretty bad coding actually)
getwd()  # returns current working directory
#setwd("~/Drobox/GitHub/Text-as-Data-Lab-Spring-2023/W1_01_26_23/")  # set working directory

# better: use the Rproject in our repo (Text-as-Data-Lab-Spring-2023.Rproj), 
# which manages the paths nicely

# 1.3 Installing and loading some useful packages

# you can always install and load packages manually with install.packages("packagename") 
# and library(packagename)
# better practice is to use pacman

if ("pacman" %in% rownames(installed.packages())==F) {
  install.packages("pacman")
}

pacman::p_load(dplyr,
               ggplot2,
               xtable,
               devtools) 

# It is also pretty bad practice to load many packages (conflicts!!!); 
# don't do it, keep things clean
# Also, you don't have to load all packages that you want to use, you can always
# use the :: to load a package in a particular instance (see next line)

# To install packages from GitHub, use:
# devtools::install_github("quanteda/quanteda.corpora")


# 1.5 Managing dependencies

# If you want to ensure that your code will run with specific package dependencies, 
# I recommend using a dependency manager for R called packrat so that you can 
# specify which version of libraries that you use.
# Find out about setting up packrat here: https://rstudio.github.io/packrat/walkthrough.html

# For R packages that are actively being developed, functions and function names 
# can change and this can break your code if you update the package but not your code! 
# (More about this next week.)

# 1.6 Loading data
polling_data  <- read.csv("W1_02_02_23/national_clinton_trump_6_20_2016.csv", 
                          stringsAsFactors = FALSE)


# ============================================================================= #
####                              WORKING WITH DATA                          ####
# ============================================================================= #

# 2.1 Take a peek, get to know the structure of the data
#############################################################

head(polling_data)  # display first lines of an object
head(polling_data, n = 10)  # same as above but specifying number of lines 
tail(polling_data)  # display last lines of an object
dim(polling_data)  # data dimensions
nrow(polling_data)  # number of rows
ncol(polling_data)  # number of columns
colnames(polling_data)  # column names
names(polling_data)  # also column names (more general command)
rownames(polling_data) # row names
class(polling_data)  # returns class of an R object
sapply(polling_data, class) # returns class for each variable (column)
str(polling_data)  # display structure of an R object (e.g. a dataframe)
glimpse(polling_data)
?sapply  # get R Documentation on this command (see Help panel below)

# 2.2 Subset dataframes 
##################################

# A) Get column with dollar sign operator
head(polling_data$Pollster)

# B) Matrix identifier: df[rowname, colname]
polling_data[, "Pollster"]

# That was pretty impossible to read in the console, let's try this:
View(polling_data[, c("Pollster", "Number.of.Observations")])
View(polling_data[polling_data$Pollster == "CBS", c("Pollster", "Number.of.Observations")])

# C) dplyr
# a very powerful package with intuitive commands for subsetting, 
# selecting, and transforming your data
# https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html  # cran documentation
# https://r4ds.had.co.nz/tidy-data.html  # going beyond dplyr, tidy data principles
# https://www.tidyverse.org  # other packages in the tidy universe (tidyverse)
# note: it's always useful to be able to do things with base R functions (helps understanding)

# Using pipe notation
polling_data %>% 
  select(Pollster) %>% 
  head(.,10)
polling_data %>% 
  select(c("Pollster", "Number.of.Observations")) %>% 
  head()

# Alternative syntax
head(select(polling_data, Pollster, Number.of.Observations)) # stick to one syntax, repetition helps recall (note order matters)


# 2.3 How to locate row(s) in a data frame 
######################################################

# A) Dollar sign operator
polling_data$Number.of.Observations[1] # Returns the first row of the data frame in the specified column (Pytho users: R indexing starts at 1)
polling_data$Number.of.Observations[1:5] # Returns the first 5 rows of the data frame in the specified column
polling_data$Number.of.Observations[polling_data$Pollster == "Quinnipiac"] # Returns all rows for the variable "Number.of.Observations" where Pollster = Quinnipiac

# B) Column name
polling_data[1, "Number.of.Observations"] 
polling_data[1:5, "Number.of.Observations"] 
polling_data[polling_data$Pollster == "Quinnipiac","Number.of.Observations"] 

# C) dplyr
# Pipe syntax
polling_data %>% 
  slice(1) %>% 
  select(Number.of.Observations) 
polling_data %>% 
  slice(1:5) %>% 
  select(Number.of.Observations)
polling_data %>% 
  filter(Pollster == "Quinnipiac") %>% 
  select(Number.of.Observations)
polling_data %>% 
  filter(Pollster == "Quinnipiac") %>% 
  slice(1) %>% 
  select(Number.of.Observations)  # can keep "piping"
polling_data %>% 
  slice(1) %>% 
  filter(Pollster == "Quinnipiac") %>% 
  select(Number.of.Observations)  # BUT note order can matter

# Alternate syntax
select(filter(polling_data, Pollster == "Quinnipiac"), Number.of.Observations)  # stick to one syntax, repetition helps recall


# 2.4 Creating new variables (columns) in a data frame 
############################################################

# A) Dollar sign operator
polling_data$net_clinton_a <- polling_data$Clinton - polling_data$Trump

polling_data <- polling_data %>% 
  relocate(Clinton, Trump, net_clinton_a)

# B) Matrix identifier
polling_data[, "net_clinton_b"]  <- polling_data[, "Clinton"] - polling_data[, "Trump"]

# C) dplyr
# Pipe syntax
polling_data <- polling_data %>% 
  mutate(net_clinton_c = Clinton - Trump)

# Alternate syntax
polling_data <- mutate(polling_data, net_clinton_d = Clinton - Trump)
#polling_data %<>% mutate(net_clinton_e = Clinton - Trump)  # requires library(magrittr) see: https://magrittr.tidyverse.org

# Are these variables equivalent to one another?
all.equal(polling_data$net_clinton_a,polling_data$net_clinton_b)  
all.equal(polling_data$net_clinton_b,polling_data$net_clinton_c) 
all.equal(polling_data$net_clinton_c,polling_data$net_clinton_d)  
#all.equal(polling_data$net_clinton_d,polling_data$net_clinton_e) 

# Yes. Yes they are.

# 2.5 Removing columns 
###############################

polling_data$net_clinton_b <- NULL
"net_clinton_b" %in% colnames(polling_data)  # one way to check if deleted column was actually deleted
polling_data[, "net_clinton_c"] <- NULL  # using matrix notation

# Using dplyr
polling_data <- polling_data %>% 
  select(-net_clinton_d)
polling_data <- polling_data %>% 
  select(-c(Source.URL, Pollster.URL))


# 2.6 Summarizing Data 
###############################

# A) Start always by getting to know the structure of the data (see above)

# B) General summary
summary(polling_data)  # summary statistics where appropriate (non-string/character variables)

# C) Single variable summary
mean(polling_data$net_clinton_a, na.rm=T)
sd(polling_data$net_clinton_a)
polling_data %>% 
  summarise(mean_net_clinton = mean(net_clinton_a))  # using dplyr
polling_data %>% 
  filter(Population == "Registered Voters") %>% 
  summarise(mean_net_clinton = mean(net_clinton_a))  # summary for a specific group

# D) Summary by group
polling_data %>% 
  group_by(Pollster) %>% 
  summarise(mean_net_clinton = mean(net_clinton_a))  # use group_by
polling_data %>% 
  group_by(Pollster) %>% 
  summarise(mean_net_clinton = mean(net_clinton_a), sd_net_clinton = sd(net_clinton_a))  # can perform multiple summary stats
table1 <- polling_data %>% 
  group_by(Pollster, Population) %>% 
  summarise(mean_net_clinton = mean(net_clinton_a)) %>% 
  ungroup %>% 
  arrange(Pollster, Population) # can group by more than one variable

View(table1)

# E) Summarizing a variable with a histogram

# Basic R graphics
hist(polling_data$net_clinton_a)

# ggplot2 graphics
plot1 <- ggplot(aes(net_clinton_a), data = polling_data) + 
  geom_histogram(bins = 15) + 
  theme_light()
plot1

# take a look at plotly for interactive plots: https://plot.ly/r/

# 2.7 Exporting data
###########################

# Exporting table to CSV
write.csv(table1,file = "W1_02_02_23/table1.csv")

# Creating LaTeX table (copy output and paste in your Latex document)
xtable(table1,caption = "Average Clinton Polling Advantage by Polling Firm")

stargazer::stargazer(table1, summary = FALSE)

# Exporting graph to pdf
pdf(width = 4, height = 3, "W1_02_02_23/plot1.pdf")
plot1
dev.off()

# or use ggsave (my preferred option)
ggsave(filename="plot1.pdf", 
       path = "W1_02_02_23", 
       width = 10, height = 6) 



# ============================================================================= #
####                              LOOPS & FUNCTIONS                          ####
# ============================================================================= #

# 3.1 For Loops
######################

for(x in names(polling_data)){ # A loop that identifies and stores variables that contain characters
  if(is.character(polling_data[, x])) {
    print(x)
  }
}

# 3.2 Apply functions (with regex)
########################################
names(polling_data)
names(polling_data) <- sapply(names(polling_data), function(i) {
  i <- gsub("\\.", "_", i) # Replaces all instances of "." with an "_"
  i <- gsub("__", "_", i) # Replaces all instances of "__" with "_"
} )
names(polling_data)

sapply(polling_data[,c("Clinton", "Trump")], mean)  # easy to apply base functions 
sapply(polling_data[,c("Clinton", "Trump", "Undecided")], 
       mean) # mean does not work with NAs
sapply(polling_data[,c("Clinton", "Trump", "Undecided")], 
       mean, na.rm = TRUE) # need to specify how to deal with NAs (this is an argument of mean)
sapply(polling_data[,c("Clinton", "Trump", "Undecided")], 
       function(x) mean(x, na.rm = TRUE)) # alternative

mean_vars1 <- sapply(polling_data[,c("Clinton", "Trump", "Undecided")], 
                     function(x) mean(x, na.rm = TRUE))  # output of sapply is a vector or a matrix
mean_vars2 <- lapply(polling_data[,c("Clinton", "Trump", "Undecided")], 
                     function(x) mean(x, na.rm = TRUE)) # output of lapply is a list

# the apply is useful when applying a fcn to rows OR columns
apply(polling_data[,c("Clinton", "Trump")], 2, mean, na.rm = TRUE) # 2 = columns
apply(polling_data[,c("Clinton", "Trump")], 1, mean, na.rm = TRUE) # 1 = rows

# dplyr version
polling_data %>% 
  summarise(avg.clinton = mean(Clinton), 
            avg.trump = mean(Trump, na.rm = TRUE))
polling_data %>% 
  rowwise() %>% 
  summarise(avg.row = mean(c(Clinton, Trump), na.rm = TRUE)) 
  
# Python users: The function passed to sapply() is the equivalent of a lambda function

# 3.3 User written functions
###################################

# Calculates the cosine similarity between two vectors
calculate_cosine_similarity <- function(vec1, vec2) { 
  nominator <- vec1 %*% vec2  # %*% specifies dot product rather than entry by entry multiplication (we could also do: sum(x * y))
  denominator <- sqrt(vec1 %*% vec1)*sqrt(vec2 %*% vec2)
  return(nominator/denominator)
}

set.seed(1984L)  # allows us to replicate result
x  <- rnorm(10) # Creates a vector of random normally distributed numbers
y  <- x*2 + 3

calculate_cosine_similarity(x,y)

# Python users: R cannot return multiple values from a function -- you will have to return a list of the values you want to return. 

calculate_distance <- function(vec1, vec2) { 
  nominator <- vec1 %*% vec2  # %*% specifies dot product rather than entry by entry multiplication (we could also do: sum(x * y))
  denominator <- sqrt(vec1 %*% vec1)*sqrt(vec2 %*% vec2)
  cos_dist <- nominator/denominator
  euc_dist <- sqrt(sum((vec1 - vec2)^2))
  return(list(cosine = cos_dist, euclidean = euc_dist))
}

calculate_distance(x,y)
dist_comp <- calculate_distance(x,y)  # we can store this result
dist_comp[["cosine"]]
dist_comp$cosine
dist_comp[[1]]

# ============================================================================= #
####                                  FINISHING UP                           ####
# ============================================================================= #

# 4.1 Save workspace after running it -- all objects, functions, etc  
# (e.g. if you have run something computationally intensive and want to save the 
# object for later use)
# Similar to pickle() in Python

save.image("workspace.RData")

# 4.2 Pick up where you left off (but note that the workspace does not include packages. 
# You need packrat for that)

rm(list = ls())

load("workspace.RData")


# ============================================================================= #
####                                FREE RESOURCES                           ####
# ============================================================================= #

# UCLA
# http://www.ats.ucla.edu/stat/r/

# Rbloggers
# https://www.r-bloggers.com/how-to-learn-r-2/

# Data Camp
# https://www.datacamp.com/

# Swirl
# http://swirlstats.com/

# If you have a question, it's probably been asked before on StackOverflow/StackExchange!

