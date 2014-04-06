# Load libraries
using DataFrames
using GLM

# Read in CSV file
nhefs = readtable("nhefs.csv")

# Get dimensions
size(nhefs)

# Preprocessing of the data
nhefs[:cens] = isna(nhefs[:wt82])
nhefs[:older] = nhefs[:age] .> 50 & !isna(nhefs[:age])

nhefs[:education_code] = 1
nhefs[:education_code][9 .< nhefs[:school] .< 11] = 2
nhefs[:education_code][nhefs[:school] .== 12] = 3
nhefs[:education_code][13 .< nhefs[:school] .< 15] = 4
nhefs[:education_code][nhefs[:school] .>= 16] = 5
nhefs[:education_code][isna(nhefs[:school])] = 6
pool!(nhefs[:education_code])

educationlabels = ["1. 8th grade or less","2. HS dropout","3. HS","4. College dropout","5. College or more","6. Unknown"]

nhefs[:education] = [educationlabels[i] for i in nhefs[:education_code]]
pool!(nhefs[:education])

# Analysis restricted to N=1566
# with non-missing values in the following covariates
# Copy original data
nhefs_original = deepcopy(nhefs)
nhefs[:id]=1:nrow(nhefs)
nhefs2=nhefs[[:id,:qsmk,:sex,:race,:age,:school,:smokeintensity,:smokeyrs,:exercise,:active,:wt71,:wt82]]
size(nhefs2)
complete_cases!(nhefs2)
size(nhefs2)

nhefs_id_matched = nhefs[findin(nhefs[:id],nhefs2[:id]),:]

# restricting data for uncensored, for comparison with observed outcome
nhefs0 = nhefs_id_matched[nhefs_id_matched[:cens] .== 0, :]
size(nhefs0)

xtabs(nhefs0[:qsmk])
pool!(nhefs0,:qsmk)
pool!(nhefs0,:sex)
pool!(nhefs0,:race)
pool!(nhefs0,:education)
pool!(nhefs0,:exercise)
pool!(nhefs0,:active)
nhefs0[:agesq] = nhefs0[:age] .^ 2
nhefs0[:smokeintensitysq] = nhefs0[:smokeintensity] .^ 2
nhefs0[:smokeyrssq] = nhefs0[:smokeyrs] .^ 2
nhefs0[:wt71sq] = nhefs0[:wt71] .^ 2
nhefs0[:qsmksi] = nhefs0[:qsmk] .* nhefs0[:smokeintensity]

glm_obj = glm(wt82_71~qsmk+sex+race+age+agesq+education+smokeintensity+smokeintensitysq+smokeyrs+smokeyrssq+exercise+active+wt71+wt71sq+qsmksi,nhefs0,Normal())

nhefs0[:meanY]=predict(glm_obj)

nhefs0[nhefs0[:seqn] .== 24770, [:meanY,:qsmk,:sex,:race,:age,:education,:smokeintensity,:smokeyrs,:exercise,:active,:wt71]]
describe(nhefs0[:meanY])
describe(nhefs0[:wt82_71])

################################################################
# PROGRAM 13.2
# Standardizing the mean outcome to the baseline confounders
# Data from Table 2.2
################################################################

id = DataArray(["Rheia", "Kronos", "Demeter", "Hades", "Hestia", "Poseidon",
  "Hera", "Zeus", "Artemis", "Apollo", "Leto", "Ares", "Athena",
  "Hephaestus", "Aphrodite", "Cyclope", "Persephone", "Hermes",
  "Hebe", "Dionysus"])
N = length(id)
L = DataArray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
A = DataArray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
Y = DataArray([0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
interv = DataArray(rep(-1, N))
data = DataFrame(L=pool(rep(L,3)),A=pool([A,rep(0,N),rep(1,N)]),Y=[Y,rep(NA,2*N)],Ydummy=rep(Y,3),interv=pool([interv,rep(0,N),rep(1,N)]),id=pool(rep(id,3)))
glm_obj2 = glm(Y~A*L,data,Normal())
predmat = ModelMatrix(ModelFrame(Ydummy~A*L,data))
data[:meanY] = predmat.m * coef(glm_obj2)

