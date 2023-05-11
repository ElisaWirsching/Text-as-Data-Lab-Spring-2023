# TA: Elisa Wirsching
# Course: Text as Data
# Date: 05/03/2023


# ============================================================================= #
####                                SETTING UP                               ####
# ============================================================================= #

# Loading packages

rm(list = ls())
set.seed(1234)

pacman::p_load(conText,
               quanteda,
               dplyr,
               text2vec,
               ggplot2)

# ============================================================================= #
####                            EMBEDDING REGRESSION                         ####
# ============================================================================= #

# great paper: https://www.cambridge.org/core/journals/american-political-science-review/article/embedding-regression-models-for-contextspecific-description-and-inference/4C90013E5C714C8483ED95CC699022FB

# what are a la carte embeddings?
# what is the purpose of embedding regressions?

# Pedro Rodriguez wrote an amazing package (conText) which allows us to implement 
# ALC embeddings and embedding regression
# We use most of his intro vignette for lab

############################
## Setup

# we need 1) a corpus, 2) pretrained embeddings, 3) a transformation matrix
# we use: the Congressional records corpus (a subset of it), the GloVe pretrained embeddings, the Khodak et al. transformation matrix

# GloVe embeddings
#glove <- readRDS(url("https://www.dropbox.com/sh/6dfr3i6no6nzvm0/AADqk6HFTvZJGNyu3FuO62kGa?dl=0/glove.rds"))
glove <- readRDS("W12_05_03_23/glove.rds")
# Khodak et al. transformation matrix
khodak <- readRDS("W12_05_03_23/khodakA.rds")

corpus <- cr_sample_corpus
docvars <- docvars(corpus) %>% 
  mutate(party_dem = ifelse(party == "D", 1, 0),
         gender_female = ifelse(gender == "F", 1, 0),
         interaction = gender_female * party_dem)
docvars(corpus) <- docvars

# tokenize corpus removing unnecessary (i.e. semantically uninformative) elements
toks <- tokens(corpus, 
               remove_punct=T, 
               remove_symbols=T, 
               remove_numbers=T, 
               remove_separators=T)

# clean out stopwords and words with 2 or fewer characters
toks_nostop <- tokens_select(toks, 
                             pattern = stopwords("en"), 
                             selection = "remove", 
                             min_nchar=3)

# only use features that appear at least 5 times in the corpus
feats <- dfm(toks_nostop, 
             tolower=T, 
             verbose = FALSE) %>% 
  dfm_trim(min_termfreq = 5) %>% 
  featnames()

# leave the pads so that non-adjacent words will not become adjacent
toks_nostop_feats <- tokens_select(toks_nostop, 
                                   feats, 
                                   padding = TRUE)




############################
## Building blocks

# suppose we are interested in differences of the term "immigration" across parties

# build a tokenized corpus of contexts surrounding the target term "immigration"
immig_toks <- tokens_context(x = toks_nostop_feats, 
                             pattern = "immigr*", 
                             window = 6L)
head(immig_toks)
length(immig_toks)
head(docvars(immig_toks), 3)

# build document-feature matrix
immig_dfm <- dfm(immig_toks)
dim(immig_dfm)
immig_dfm[1:3,]

# build a document-embedding-matrix
# We embed a document by multiplying each of it’s feature counts with their 
# corresponding pre-trained feature-embeddings, column-averaging the resulting vectors, 
# and multiplying by the transformation matrix.
immig_dem <- dem(x = immig_dfm, 
                 pre_trained = glove, 
                 transform = TRUE, 
                 transform_matrix = khodak, 
                 verbose = TRUE)
dim(immig_dem)
head(immig_dem)



############################
## ALC embedding vectors

# We now have an ALC embedding for each instance of “immigration” in our sample corpus
# To get a single corpus-wide ALC embedding for “immigration”, 
# we can simply take the column-average of the single-instance ALC embeddings

# to get a single "corpus-wide" embedding, take the column average
immig_wv <- matrix(colMeans(immig_dem), 
                   ncol = ncol(immig_dem)) %>%  `rownames<-`("immigration")
dim(immig_wv)
immig_wv

# to get group-specific embeddings, average within party
immig_wv_party <- dem_group(immig_dem, 
                            groups = immig_dem@docvars$party)
dim(immig_wv_party)
immig_wv_party


############################
## Comparing across groups

# nearest neighbors
# ------------------------
# find nearest neighbors by party
# setting as_list = FALSE combines each group's results into a single tibble (useful for joint plotting)
immig_nns <- nns(immig_wv_party, 
                 pre_trained = glove, 
                 N = 5, 
                 candidates = immig_wv_party@features, 
                 as_list = TRUE)

# check out results for party
immig_nns[["R"]]
immig_nns[["D"]]


# Cosine similarity
# --------------------
# compute the cosine similarity between each party's embedding and a specific set of features
cos_sim(immig_wv_party, 
        pre_trained = glove, 
        features = c('reform', 'enforcement'), 
        as_list = FALSE)


# cosine similarity ratio

# we limit candidates to features in our corpus
feats <- featnames(dfm(immig_toks))

# compute ratio
set.seed(2021L)
immig_nns_ratio <- get_nns_ratio(x = immig_toks, 
                                 N = 10,
                                 groups = docvars(immig_toks, 'party'),
                                 numerator = "R",
                                 candidates = feats,
                                 pre_trained = glove,
                                 transform = TRUE,
                                 transform_matrix = khodak,
                                 bootstrap = TRUE,
                                 num_bootstraps = 100,
                                 permute = TRUE,
                                 num_permutations = 10,
                                 verbose = FALSE)

plot_nns_ratio(x = immig_nns_ratio, alpha = 0.01, horizontal = F)



# Embedding regression
# ------------------------

# conText() tries to follow a similar syntax as R’s lm() and glm() functions. 
# data must be a quanteda tokens object with covariates stored as document variables (docvars). 
# We next specify a formula consisting of the target word of interest, e.g. “immigration” and the set of covariates. 
# To use all covariates in data, we can speficy immigration ~ .. formula can also take vectors of target words 
# e.g. c("immigration", "immigrants") ~ party + gender and phrases e.g. "immigration reform" ~ party + gender 
# place phrases in quotation marks


set.seed(2021L)
model1 <- conText(formula = immigration ~ party + gender,
                  data = toks_nostop_feats,
                  pre_trained = glove,
                  transform = TRUE, transform_matrix = khodak,
                  bootstrap = TRUE, num_bootstraps = 100,
                  permute = TRUE, num_permutations = 100,
                  window = 6, case_insensitive = TRUE,
                  verbose = FALSE)

model1@normed_coefficients

# D-dimensional beta coefficients
# the intercept in this case is the ALC embedding for female Democrats
# beta coefficients can be combined to get each group's ALC embedding
DF_wv <- model1['(Intercept)',] # (D)emocrat - (F)emale 
DM_wv <- model1['(Intercept)',] + model1['gender_M',] # (D)emocrat - (M)ale 
RF_wv <- model1['(Intercept)',] + model1['party_R',]  # (R)epublican - (F)emale 
RM_wv <- model1['(Intercept)',] + model1['party_R',] + model1['gender_M',] # (R)epublican - (M)ale 

# nearest neighbors
nns(rbind(DF_wv,DM_wv), N = 10, pre_trained = glove, candidates = model1@features)



# model with interaction also possible, but need to calculate interaction ourselves
model2 <- conText(formula = immigration ~ party_dem + gender_female + interaction,
                  data = toks_nostop_feats,
                  pre_trained = glove,
                  transform = TRUE, transform_matrix = khodak,
                  bootstrap = TRUE, num_bootstraps = 100,
                  permute = TRUE, num_permutations = 100,
                  window = 6, case_insensitive = TRUE,
                  verbose = FALSE)

ggplot(data = model2@normed_coefficients,
       aes(x = factor(coefficient, levels = c("gender_female", "party_dem", "interaction")), y = normed.estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = normed.estimate - 1.96*std.error,
                    ymax = normed.estimate + 1.96*std.error),
                width = 0.5) +
  labs(x = "") +
  theme_bw()

DF_wv <- model2['(Intercept)',] + model2['gender_female',] + model2['party_dem',] + model2['interaction',]# (D)emocrat - (F)emale 
DM_wv <- model2['(Intercept)',] + model2['party_dem',] # (D)emocrat - (M)ale 
RF_wv <- model2['(Intercept)',] + model2['gender_female',]  # (R)epublican - (F)emale 
RM_wv <- model2['(Intercept)',] # (R)epublican - (M)ale 

# nearest neighbors
nns(rbind(DF_wv,RM_wv), N = 10, pre_trained = glove, candidates = model2@features)

