??9
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??5
?
'x_deep_fm_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*8
shared_name)'x_deep_fm_1/batch_normalization_1/gamma
?
;x_deep_fm_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'x_deep_fm_1/batch_normalization_1/gamma*
_output_shapes
:'*
dtype0
?
&x_deep_fm_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*7
shared_name(&x_deep_fm_1/batch_normalization_1/beta
?
:x_deep_fm_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&x_deep_fm_1/batch_normalization_1/beta*
_output_shapes
:'*
dtype0
?
-x_deep_fm_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*>
shared_name/-x_deep_fm_1/batch_normalization_1/moving_mean
?
Ax_deep_fm_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-x_deep_fm_1/batch_normalization_1/moving_mean*
_output_shapes
:'*
dtype0
?
1x_deep_fm_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*B
shared_name31x_deep_fm_1/batch_normalization_1/moving_variance
?
Ex_deep_fm_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1x_deep_fm_1/batch_normalization_1/moving_variance*
_output_shapes
:'*
dtype0
?
x_deep_fm_1/cin_1/w0VarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_namex_deep_fm_1/cin_1/w0
?
(x_deep_fm_1/cin_1/w0/Read/ReadVariableOpReadVariableOpx_deep_fm_1/cin_1/w0*#
_output_shapes
:? *
dtype0
?
x_deep_fm_1/cin_1/w1VarHandleOp*
_output_shapes
: *
dtype0*
shape:? *%
shared_namex_deep_fm_1/cin_1/w1
?
(x_deep_fm_1/cin_1/w1/Read/ReadVariableOpReadVariableOpx_deep_fm_1/cin_1/w1*#
_output_shapes
:? *
dtype0
?
x_deep_fm_1/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*+
shared_namex_deep_fm_1/dense_9/kernel
?
.x_deep_fm_1/dense_9/kernel/Read/ReadVariableOpReadVariableOpx_deep_fm_1/dense_9/kernel*
_output_shapes

:@*
dtype0
?
x_deep_fm_1/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namex_deep_fm_1/dense_9/bias
?
,x_deep_fm_1/dense_9/bias/Read/ReadVariableOpReadVariableOpx_deep_fm_1/dense_9/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
#x_deep_fm_1/embedding_26/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*4
shared_name%#x_deep_fm_1/embedding_26/embeddings
?
7x_deep_fm_1/embedding_26/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_26/embeddings*
_output_shapes

:h*
dtype0
?
#x_deep_fm_1/embedding_27/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_27/embeddings
?
7x_deep_fm_1/embedding_27/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_27/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_28/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_28/embeddings
?
7x_deep_fm_1/embedding_28/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_28/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_29/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_29/embeddings
?
7x_deep_fm_1/embedding_29/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_29/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_30/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**4
shared_name%#x_deep_fm_1/embedding_30/embeddings
?
7x_deep_fm_1/embedding_30/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_30/embeddings*
_output_shapes

:**
dtype0
?
#x_deep_fm_1/embedding_31/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*4
shared_name%#x_deep_fm_1/embedding_31/embeddings
?
7x_deep_fm_1/embedding_31/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_31/embeddings*
_output_shapes

:	*
dtype0
?
#x_deep_fm_1/embedding_32/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_32/embeddings
?
7x_deep_fm_1/embedding_32/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_32/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_33/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*4
shared_name%#x_deep_fm_1/embedding_33/embeddings
?
7x_deep_fm_1/embedding_33/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_33/embeddings*
_output_shapes

:C*
dtype0
?
#x_deep_fm_1/embedding_34/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#x_deep_fm_1/embedding_34/embeddings
?
7x_deep_fm_1/embedding_34/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_34/embeddings*
_output_shapes

:*
dtype0
?
#x_deep_fm_1/embedding_35/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_35/embeddings
?
7x_deep_fm_1/embedding_35/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_35/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_36/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_36/embeddings
?
7x_deep_fm_1/embedding_36/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_36/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_37/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_37/embeddings
?
7x_deep_fm_1/embedding_37/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_37/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_38/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_38/embeddings
?
7x_deep_fm_1/embedding_38/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_38/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_39/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#x_deep_fm_1/embedding_39/embeddings
?
7x_deep_fm_1/embedding_39/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_39/embeddings*
_output_shapes

:*
dtype0
?
#x_deep_fm_1/embedding_40/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*4
shared_name%#x_deep_fm_1/embedding_40/embeddings
?
7x_deep_fm_1/embedding_40/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_40/embeddings*
_output_shapes
:	?
*
dtype0
?
#x_deep_fm_1/embedding_41/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_41/embeddings
?
7x_deep_fm_1/embedding_41/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_41/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_42/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*4
shared_name%#x_deep_fm_1/embedding_42/embeddings
?
7x_deep_fm_1/embedding_42/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_42/embeddings*
_output_shapes

:
*
dtype0
?
#x_deep_fm_1/embedding_43/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_43/embeddings
?
7x_deep_fm_1/embedding_43/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_43/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_44/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_44/embeddings
?
7x_deep_fm_1/embedding_44/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_44/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_45/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#x_deep_fm_1/embedding_45/embeddings
?
7x_deep_fm_1/embedding_45/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_45/embeddings*
_output_shapes

:*
dtype0
?
#x_deep_fm_1/embedding_46/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_46/embeddings
?
7x_deep_fm_1/embedding_46/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_46/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_47/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*4
shared_name%#x_deep_fm_1/embedding_47/embeddings
?
7x_deep_fm_1/embedding_47/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_47/embeddings*
_output_shapes

:
*
dtype0
?
#x_deep_fm_1/embedding_48/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#x_deep_fm_1/embedding_48/embeddings
?
7x_deep_fm_1/embedding_48/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_48/embeddings*
_output_shapes

:*
dtype0
?
#x_deep_fm_1/embedding_49/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_49/embeddings
?
7x_deep_fm_1/embedding_49/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_49/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/embedding_50/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*4
shared_name%#x_deep_fm_1/embedding_50/embeddings
?
7x_deep_fm_1/embedding_50/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_50/embeddings*
_output_shapes

:'*
dtype0
?
#x_deep_fm_1/embedding_51/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*4
shared_name%#x_deep_fm_1/embedding_51/embeddings
?
7x_deep_fm_1/embedding_51/embeddings/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/embedding_51/embeddings*
_output_shapes
:	?*
dtype0
?
#x_deep_fm_1/linear_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*4
shared_name%#x_deep_fm_1/linear_1/dense_5/kernel
?
7x_deep_fm_1/linear_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp#x_deep_fm_1/linear_1/dense_5/kernel*
_output_shapes

:'*
dtype0
?
!x_deep_fm_1/linear_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!x_deep_fm_1/linear_1/dense_5/bias
?
5x_deep_fm_1/linear_1/dense_5/bias/Read/ReadVariableOpReadVariableOp!x_deep_fm_1/linear_1/dense_5/bias*
_output_shapes
:*
dtype0
?
(x_deep_fm_1/dense_layer_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*9
shared_name*(x_deep_fm_1/dense_layer_1/dense_6/kernel
?
<x_deep_fm_1/dense_layer_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp(x_deep_fm_1/dense_layer_1/dense_6/kernel*
_output_shapes
:	?@*
dtype0
?
&x_deep_fm_1/dense_layer_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&x_deep_fm_1/dense_layer_1/dense_6/bias
?
:x_deep_fm_1/dense_layer_1/dense_6/bias/Read/ReadVariableOpReadVariableOp&x_deep_fm_1/dense_layer_1/dense_6/bias*
_output_shapes
:@*
dtype0
?
(x_deep_fm_1/dense_layer_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*9
shared_name*(x_deep_fm_1/dense_layer_1/dense_7/kernel
?
<x_deep_fm_1/dense_layer_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp(x_deep_fm_1/dense_layer_1/dense_7/kernel*
_output_shapes

:@@*
dtype0
?
&x_deep_fm_1/dense_layer_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&x_deep_fm_1/dense_layer_1/dense_7/bias
?
:x_deep_fm_1/dense_layer_1/dense_7/bias/Read/ReadVariableOpReadVariableOp&x_deep_fm_1/dense_layer_1/dense_7/bias*
_output_shapes
:@*
dtype0
?
(x_deep_fm_1/dense_layer_1/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*9
shared_name*(x_deep_fm_1/dense_layer_1/dense_8/kernel
?
<x_deep_fm_1/dense_layer_1/dense_8/kernel/Read/ReadVariableOpReadVariableOp(x_deep_fm_1/dense_layer_1/dense_8/kernel*
_output_shapes

:@*
dtype0
?
&x_deep_fm_1/dense_layer_1/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&x_deep_fm_1/dense_layer_1/dense_8/bias
?
:x_deep_fm_1/dense_layer_1/dense_8/bias/Read/ReadVariableOpReadVariableOp&x_deep_fm_1/dense_layer_1/dense_8/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
?
.Adam/x_deep_fm_1/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*?
shared_name0.Adam/x_deep_fm_1/batch_normalization_1/gamma/m
?
BAdam/x_deep_fm_1/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp.Adam/x_deep_fm_1/batch_normalization_1/gamma/m*
_output_shapes
:'*
dtype0
?
-Adam/x_deep_fm_1/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*>
shared_name/-Adam/x_deep_fm_1/batch_normalization_1/beta/m
?
AAdam/x_deep_fm_1/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/batch_normalization_1/beta/m*
_output_shapes
:'*
dtype0
?
Adam/x_deep_fm_1/cin_1/w0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *,
shared_nameAdam/x_deep_fm_1/cin_1/w0/m
?
/Adam/x_deep_fm_1/cin_1/w0/m/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/cin_1/w0/m*#
_output_shapes
:? *
dtype0
?
Adam/x_deep_fm_1/cin_1/w1/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *,
shared_nameAdam/x_deep_fm_1/cin_1/w1/m
?
/Adam/x_deep_fm_1/cin_1/w1/m/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/cin_1/w1/m*#
_output_shapes
:? *
dtype0
?
!Adam/x_deep_fm_1/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/x_deep_fm_1/dense_9/kernel/m
?
5Adam/x_deep_fm_1/dense_9/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/x_deep_fm_1/dense_9/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/x_deep_fm_1/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/x_deep_fm_1/dense_9/bias/m
?
3Adam/x_deep_fm_1/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/dense_9/bias/m*
_output_shapes
:*
dtype0
?
*Adam/x_deep_fm_1/embedding_26/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*;
shared_name,*Adam/x_deep_fm_1/embedding_26/embeddings/m
?
>Adam/x_deep_fm_1/embedding_26/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_26/embeddings/m*
_output_shapes

:h*
dtype0
?
*Adam/x_deep_fm_1/embedding_27/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_27/embeddings/m
?
>Adam/x_deep_fm_1/embedding_27/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_27/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_28/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_28/embeddings/m
?
>Adam/x_deep_fm_1/embedding_28/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_28/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_29/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_29/embeddings/m
?
>Adam/x_deep_fm_1/embedding_29/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_29/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_30/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**;
shared_name,*Adam/x_deep_fm_1/embedding_30/embeddings/m
?
>Adam/x_deep_fm_1/embedding_30/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_30/embeddings/m*
_output_shapes

:**
dtype0
?
*Adam/x_deep_fm_1/embedding_31/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*;
shared_name,*Adam/x_deep_fm_1/embedding_31/embeddings/m
?
>Adam/x_deep_fm_1/embedding_31/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_31/embeddings/m*
_output_shapes

:	*
dtype0
?
*Adam/x_deep_fm_1/embedding_32/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_32/embeddings/m
?
>Adam/x_deep_fm_1/embedding_32/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_32/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_33/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*;
shared_name,*Adam/x_deep_fm_1/embedding_33/embeddings/m
?
>Adam/x_deep_fm_1/embedding_33/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_33/embeddings/m*
_output_shapes

:C*
dtype0
?
*Adam/x_deep_fm_1/embedding_34/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_34/embeddings/m
?
>Adam/x_deep_fm_1/embedding_34/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_34/embeddings/m*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_35/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_35/embeddings/m
?
>Adam/x_deep_fm_1/embedding_35/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_35/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_36/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_36/embeddings/m
?
>Adam/x_deep_fm_1/embedding_36/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_36/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_37/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_37/embeddings/m
?
>Adam/x_deep_fm_1/embedding_37/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_37/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_38/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_38/embeddings/m
?
>Adam/x_deep_fm_1/embedding_38/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_38/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_39/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_39/embeddings/m
?
>Adam/x_deep_fm_1/embedding_39/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_39/embeddings/m*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_40/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*;
shared_name,*Adam/x_deep_fm_1/embedding_40/embeddings/m
?
>Adam/x_deep_fm_1/embedding_40/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_40/embeddings/m*
_output_shapes
:	?
*
dtype0
?
*Adam/x_deep_fm_1/embedding_41/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_41/embeddings/m
?
>Adam/x_deep_fm_1/embedding_41/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_41/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_42/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*Adam/x_deep_fm_1/embedding_42/embeddings/m
?
>Adam/x_deep_fm_1/embedding_42/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_42/embeddings/m*
_output_shapes

:
*
dtype0
?
*Adam/x_deep_fm_1/embedding_43/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_43/embeddings/m
?
>Adam/x_deep_fm_1/embedding_43/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_43/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_44/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_44/embeddings/m
?
>Adam/x_deep_fm_1/embedding_44/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_44/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_45/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_45/embeddings/m
?
>Adam/x_deep_fm_1/embedding_45/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_45/embeddings/m*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_46/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_46/embeddings/m
?
>Adam/x_deep_fm_1/embedding_46/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_46/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_47/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*Adam/x_deep_fm_1/embedding_47/embeddings/m
?
>Adam/x_deep_fm_1/embedding_47/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_47/embeddings/m*
_output_shapes

:
*
dtype0
?
*Adam/x_deep_fm_1/embedding_48/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_48/embeddings/m
?
>Adam/x_deep_fm_1/embedding_48/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_48/embeddings/m*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_49/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_49/embeddings/m
?
>Adam/x_deep_fm_1/embedding_49/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_49/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_50/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*;
shared_name,*Adam/x_deep_fm_1/embedding_50/embeddings/m
?
>Adam/x_deep_fm_1/embedding_50/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_50/embeddings/m*
_output_shapes

:'*
dtype0
?
*Adam/x_deep_fm_1/embedding_51/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_51/embeddings/m
?
>Adam/x_deep_fm_1/embedding_51/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_51/embeddings/m*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/linear_1/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*;
shared_name,*Adam/x_deep_fm_1/linear_1/dense_5/kernel/m
?
>Adam/x_deep_fm_1/linear_1/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/linear_1/dense_5/kernel/m*
_output_shapes

:'*
dtype0
?
(Adam/x_deep_fm_1/linear_1/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/x_deep_fm_1/linear_1/dense_5/bias/m
?
<Adam/x_deep_fm_1/linear_1/dense_5/bias/m/Read/ReadVariableOpReadVariableOp(Adam/x_deep_fm_1/linear_1/dense_5/bias/m*
_output_shapes
:*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m
?
CAdam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m*
_output_shapes
:	?@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/m
?
AAdam/x_deep_fm_1/dense_layer_1/dense_6/bias/m/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/m*
_output_shapes
:@*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m
?
CAdam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m*
_output_shapes

:@@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/m
?
AAdam/x_deep_fm_1/dense_layer_1/dense_7/bias/m/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/m*
_output_shapes
:@*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m
?
CAdam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m*
_output_shapes

:@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/m
?
AAdam/x_deep_fm_1/dense_layer_1/dense_8/bias/m/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/m*
_output_shapes
:*
dtype0
?
.Adam/x_deep_fm_1/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*?
shared_name0.Adam/x_deep_fm_1/batch_normalization_1/gamma/v
?
BAdam/x_deep_fm_1/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp.Adam/x_deep_fm_1/batch_normalization_1/gamma/v*
_output_shapes
:'*
dtype0
?
-Adam/x_deep_fm_1/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:'*>
shared_name/-Adam/x_deep_fm_1/batch_normalization_1/beta/v
?
AAdam/x_deep_fm_1/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/batch_normalization_1/beta/v*
_output_shapes
:'*
dtype0
?
Adam/x_deep_fm_1/cin_1/w0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *,
shared_nameAdam/x_deep_fm_1/cin_1/w0/v
?
/Adam/x_deep_fm_1/cin_1/w0/v/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/cin_1/w0/v*#
_output_shapes
:? *
dtype0
?
Adam/x_deep_fm_1/cin_1/w1/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *,
shared_nameAdam/x_deep_fm_1/cin_1/w1/v
?
/Adam/x_deep_fm_1/cin_1/w1/v/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/cin_1/w1/v*#
_output_shapes
:? *
dtype0
?
!Adam/x_deep_fm_1/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!Adam/x_deep_fm_1/dense_9/kernel/v
?
5Adam/x_deep_fm_1/dense_9/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/x_deep_fm_1/dense_9/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/x_deep_fm_1/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/x_deep_fm_1/dense_9/bias/v
?
3Adam/x_deep_fm_1/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/x_deep_fm_1/dense_9/bias/v*
_output_shapes
:*
dtype0
?
*Adam/x_deep_fm_1/embedding_26/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*;
shared_name,*Adam/x_deep_fm_1/embedding_26/embeddings/v
?
>Adam/x_deep_fm_1/embedding_26/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_26/embeddings/v*
_output_shapes

:h*
dtype0
?
*Adam/x_deep_fm_1/embedding_27/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_27/embeddings/v
?
>Adam/x_deep_fm_1/embedding_27/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_27/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_28/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_28/embeddings/v
?
>Adam/x_deep_fm_1/embedding_28/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_28/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_29/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_29/embeddings/v
?
>Adam/x_deep_fm_1/embedding_29/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_29/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_30/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**;
shared_name,*Adam/x_deep_fm_1/embedding_30/embeddings/v
?
>Adam/x_deep_fm_1/embedding_30/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_30/embeddings/v*
_output_shapes

:**
dtype0
?
*Adam/x_deep_fm_1/embedding_31/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*;
shared_name,*Adam/x_deep_fm_1/embedding_31/embeddings/v
?
>Adam/x_deep_fm_1/embedding_31/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_31/embeddings/v*
_output_shapes

:	*
dtype0
?
*Adam/x_deep_fm_1/embedding_32/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_32/embeddings/v
?
>Adam/x_deep_fm_1/embedding_32/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_32/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_33/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*;
shared_name,*Adam/x_deep_fm_1/embedding_33/embeddings/v
?
>Adam/x_deep_fm_1/embedding_33/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_33/embeddings/v*
_output_shapes

:C*
dtype0
?
*Adam/x_deep_fm_1/embedding_34/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_34/embeddings/v
?
>Adam/x_deep_fm_1/embedding_34/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_34/embeddings/v*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_35/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_35/embeddings/v
?
>Adam/x_deep_fm_1/embedding_35/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_35/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_36/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_36/embeddings/v
?
>Adam/x_deep_fm_1/embedding_36/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_36/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_37/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_37/embeddings/v
?
>Adam/x_deep_fm_1/embedding_37/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_37/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_38/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_38/embeddings/v
?
>Adam/x_deep_fm_1/embedding_38/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_38/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_39/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_39/embeddings/v
?
>Adam/x_deep_fm_1/embedding_39/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_39/embeddings/v*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_40/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*;
shared_name,*Adam/x_deep_fm_1/embedding_40/embeddings/v
?
>Adam/x_deep_fm_1/embedding_40/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_40/embeddings/v*
_output_shapes
:	?
*
dtype0
?
*Adam/x_deep_fm_1/embedding_41/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_41/embeddings/v
?
>Adam/x_deep_fm_1/embedding_41/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_41/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_42/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*Adam/x_deep_fm_1/embedding_42/embeddings/v
?
>Adam/x_deep_fm_1/embedding_42/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_42/embeddings/v*
_output_shapes

:
*
dtype0
?
*Adam/x_deep_fm_1/embedding_43/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_43/embeddings/v
?
>Adam/x_deep_fm_1/embedding_43/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_43/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_44/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_44/embeddings/v
?
>Adam/x_deep_fm_1/embedding_44/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_44/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_45/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_45/embeddings/v
?
>Adam/x_deep_fm_1/embedding_45/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_45/embeddings/v*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_46/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_46/embeddings/v
?
>Adam/x_deep_fm_1/embedding_46/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_46/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_47/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*;
shared_name,*Adam/x_deep_fm_1/embedding_47/embeddings/v
?
>Adam/x_deep_fm_1/embedding_47/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_47/embeddings/v*
_output_shapes

:
*
dtype0
?
*Adam/x_deep_fm_1/embedding_48/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adam/x_deep_fm_1/embedding_48/embeddings/v
?
>Adam/x_deep_fm_1/embedding_48/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_48/embeddings/v*
_output_shapes

:*
dtype0
?
*Adam/x_deep_fm_1/embedding_49/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_49/embeddings/v
?
>Adam/x_deep_fm_1/embedding_49/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_49/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/embedding_50/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*;
shared_name,*Adam/x_deep_fm_1/embedding_50/embeddings/v
?
>Adam/x_deep_fm_1/embedding_50/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_50/embeddings/v*
_output_shapes

:'*
dtype0
?
*Adam/x_deep_fm_1/embedding_51/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*;
shared_name,*Adam/x_deep_fm_1/embedding_51/embeddings/v
?
>Adam/x_deep_fm_1/embedding_51/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/embedding_51/embeddings/v*
_output_shapes
:	?*
dtype0
?
*Adam/x_deep_fm_1/linear_1/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*;
shared_name,*Adam/x_deep_fm_1/linear_1/dense_5/kernel/v
?
>Adam/x_deep_fm_1/linear_1/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp*Adam/x_deep_fm_1/linear_1/dense_5/kernel/v*
_output_shapes

:'*
dtype0
?
(Adam/x_deep_fm_1/linear_1/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/x_deep_fm_1/linear_1/dense_5/bias/v
?
<Adam/x_deep_fm_1/linear_1/dense_5/bias/v/Read/ReadVariableOpReadVariableOp(Adam/x_deep_fm_1/linear_1/dense_5/bias/v*
_output_shapes
:*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v
?
CAdam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v*
_output_shapes
:	?@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/v
?
AAdam/x_deep_fm_1/dense_layer_1/dense_6/bias/v/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/v*
_output_shapes
:@*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v
?
CAdam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v*
_output_shapes

:@@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/v
?
AAdam/x_deep_fm_1/dense_layer_1/dense_7/bias/v/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/v*
_output_shapes
:@*
dtype0
?
/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*@
shared_name1/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v
?
CAdam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v*
_output_shapes

:@*
dtype0
?
-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/v
?
AAdam/x_deep_fm_1/dense_layer_1/dense_8/bias/v/Read/ReadVariableOpReadVariableOp-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
dense_feature_columns
sparse_feature_columns
embed_layers
bn

linear
dense_layer
	cin_layer
	out_layer
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
^
0
1
2
3
4
5
6
7
8
9
10
11
12
?
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13
*14
+15
,16
-17
.18
/19
020
121
222
323
424
525
?
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13
D14
E15
F16
G17
H18
I19
J20
K21
L22
M23
N24
O25
?
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
a
Y	out_layer
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?
^hidden_layers
_	out_layer
`dropout
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?
ecin_size
f	field_num
gw0
hw1
	icin_W
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
h

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rateQm?Rm?gm?hm?nm?om?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Qv?Rv?gv?hv?nv?ov?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
y0
z1
{2
|3
}4
~5
6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
Q26
R27
S28
T29
?30
?31
?32
?33
?34
?35
?36
?37
g38
h39
n40
o41
?
y0
z1
{2
|3
}4
~5
6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
Q26
R27
?28
?29
?30
?31
?32
?33
?34
?35
g36
h37
n38
o39
 
?

	variables
trainable_variables
regularization_losses
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?non_trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
f
y
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
z
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
{
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
|
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
}
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f
~
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
f

embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
g
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
 
`^
VARIABLE_VALUE'x_deep_fm_1/batch_normalization_1/gamma#bn/gamma/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE&x_deep_fm_1/batch_normalization_1/beta"bn/beta/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE-x_deep_fm_1/batch_normalization_1/moving_mean)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE1x_deep_fm_1/batch_normalization_1/moving_variance-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
S2
T3

Q0
R1
 
?
U	variables
Vtrainable_variables
Wregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?0
?1

?0
?1
 
?
Z	variables
[trainable_variables
\regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0
?1
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
0
?0
?1
?2
?3
?4
?5
0
?0
?1
?2
?3
?4
?5
 
?
a	variables
btrainable_variables
cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
 
 
QO
VARIABLE_VALUEx_deep_fm_1/cin_1/w0'cin_layer/w0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEx_deep_fm_1/cin_1/w1'cin_layer/w1/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1

g0
h1
 
?
j	variables
ktrainable_variables
lregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
[Y
VARIABLE_VALUEx_deep_fm_1/dense_9/kernel+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEx_deep_fm_1/dense_9/bias)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

n0
o1
 
?
p	variables
qtrainable_variables
rregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_26/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_27/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_28/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_29/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_30/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_31/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_32/embeddings&variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_33/embeddings&variables/7/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_34/embeddings&variables/8/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#x_deep_fm_1/embedding_35/embeddings&variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_36/embeddings'variables/10/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_37/embeddings'variables/11/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_38/embeddings'variables/12/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_39/embeddings'variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_40/embeddings'variables/14/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_41/embeddings'variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_42/embeddings'variables/16/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_43/embeddings'variables/17/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_44/embeddings'variables/18/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_45/embeddings'variables/19/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_46/embeddings'variables/20/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_47/embeddings'variables/21/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_48/embeddings'variables/22/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_49/embeddings'variables/23/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_50/embeddings'variables/24/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/embedding_51/embeddings'variables/25/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#x_deep_fm_1/linear_1/dense_5/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!x_deep_fm_1/linear_1/dense_5/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(x_deep_fm_1/dense_layer_1/dense_6/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&x_deep_fm_1/dense_layer_1/dense_6/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(x_deep_fm_1/dense_layer_1/dense_7/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&x_deep_fm_1/dense_layer_1/dense_7/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(x_deep_fm_1/dense_layer_1/dense_8/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&x_deep_fm_1/dense_layer_1/dense_8/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
 
 
?
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13
D14
E15
F16
G17
H18
I19
J20
K21
L22
M23
N24
O25
26
27
28
29
30

?0
?1
?2

S0
T1

y0

y0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

z0

z0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

{0

{0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

|0

|0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

}0

}0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

~0

~0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

0

0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0

?0
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

S0
T1
 
 
 
 

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
 
 
 
 

Y0
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
 
 
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
 
 
 
 

?0
?1
_2
`3
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
\
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers

?0
?1

?0
?1
 
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
??
VARIABLE_VALUE.Adam/x_deep_fm_1/batch_normalization_1/gamma/m?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE-Adam/x_deep_fm_1/batch_normalization_1/beta/m>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/x_deep_fm_1/cin_1/w0/mCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/x_deep_fm_1/cin_1/w1/mCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE!Adam/x_deep_fm_1/dense_9/kernel/mGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/x_deep_fm_1/dense_9/bias/mEout_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_26/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_27/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_28/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_29/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_30/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_31/embeddings/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_32/embeddings/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_33/embeddings/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_34/embeddings/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_35/embeddings/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_36/embeddings/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_37/embeddings/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_38/embeddings/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_39/embeddings/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_40/embeddings/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_41/embeddings/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_42/embeddings/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_43/embeddings/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_44/embeddings/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_45/embeddings/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_46/embeddings/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_47/embeddings/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_48/embeddings/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_49/embeddings/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_50/embeddings/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_51/embeddings/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/linear_1/dense_5/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/x_deep_fm_1/linear_1/dense_5/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/x_deep_fm_1/batch_normalization_1/gamma/v?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE-Adam/x_deep_fm_1/batch_normalization_1/beta/v>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/x_deep_fm_1/cin_1/w0/vCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/x_deep_fm_1/cin_1/w1/vCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE!Adam/x_deep_fm_1/dense_9/kernel/vGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/x_deep_fm_1/dense_9/bias/vEout_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_26/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_27/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_28/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_29/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_30/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_31/embeddings/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_32/embeddings/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_33/embeddings/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_34/embeddings/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_35/embeddings/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_36/embeddings/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_37/embeddings/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_38/embeddings/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_39/embeddings/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_40/embeddings/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_41/embeddings/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_42/embeddings/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_43/embeddings/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_44/embeddings/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_45/embeddings/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_46/embeddings/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_47/embeddings/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_48/embeddings/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_49/embeddings/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_50/embeddings/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/embedding_51/embeddings/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/x_deep_fm_1/linear_1/dense_5/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUE(Adam/x_deep_fm_1/linear_1/dense_5/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????'*
dtype0*
shape:?????????'
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-x_deep_fm_1/batch_normalization_1/moving_mean1x_deep_fm_1/batch_normalization_1/moving_variance&x_deep_fm_1/batch_normalization_1/beta'x_deep_fm_1/batch_normalization_1/gamma#x_deep_fm_1/linear_1/dense_5/kernel!x_deep_fm_1/linear_1/dense_5/bias#x_deep_fm_1/embedding_26/embeddings#x_deep_fm_1/embedding_27/embeddings#x_deep_fm_1/embedding_28/embeddings#x_deep_fm_1/embedding_29/embeddings#x_deep_fm_1/embedding_30/embeddings#x_deep_fm_1/embedding_31/embeddings#x_deep_fm_1/embedding_32/embeddings#x_deep_fm_1/embedding_33/embeddings#x_deep_fm_1/embedding_34/embeddings#x_deep_fm_1/embedding_35/embeddings#x_deep_fm_1/embedding_36/embeddings#x_deep_fm_1/embedding_37/embeddings#x_deep_fm_1/embedding_38/embeddings#x_deep_fm_1/embedding_39/embeddings#x_deep_fm_1/embedding_40/embeddings#x_deep_fm_1/embedding_41/embeddings#x_deep_fm_1/embedding_42/embeddings#x_deep_fm_1/embedding_43/embeddings#x_deep_fm_1/embedding_44/embeddings#x_deep_fm_1/embedding_45/embeddings#x_deep_fm_1/embedding_46/embeddings#x_deep_fm_1/embedding_47/embeddings#x_deep_fm_1/embedding_48/embeddings#x_deep_fm_1/embedding_49/embeddings#x_deep_fm_1/embedding_50/embeddings#x_deep_fm_1/embedding_51/embeddingsx_deep_fm_1/cin_1/w0x_deep_fm_1/cin_1/w1(x_deep_fm_1/dense_layer_1/dense_6/kernel&x_deep_fm_1/dense_layer_1/dense_6/bias(x_deep_fm_1/dense_layer_1/dense_7/kernel&x_deep_fm_1/dense_layer_1/dense_7/bias(x_deep_fm_1/dense_layer_1/dense_8/kernel&x_deep_fm_1/dense_layer_1/dense_8/biasx_deep_fm_1/dense_9/kernelx_deep_fm_1/dense_9/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_87748
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?A
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;x_deep_fm_1/batch_normalization_1/gamma/Read/ReadVariableOp:x_deep_fm_1/batch_normalization_1/beta/Read/ReadVariableOpAx_deep_fm_1/batch_normalization_1/moving_mean/Read/ReadVariableOpEx_deep_fm_1/batch_normalization_1/moving_variance/Read/ReadVariableOp(x_deep_fm_1/cin_1/w0/Read/ReadVariableOp(x_deep_fm_1/cin_1/w1/Read/ReadVariableOp.x_deep_fm_1/dense_9/kernel/Read/ReadVariableOp,x_deep_fm_1/dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp7x_deep_fm_1/embedding_26/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_27/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_28/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_29/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_30/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_31/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_32/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_33/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_34/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_35/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_36/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_37/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_38/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_39/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_40/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_41/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_42/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_43/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_44/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_45/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_46/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_47/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_48/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_49/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_50/embeddings/Read/ReadVariableOp7x_deep_fm_1/embedding_51/embeddings/Read/ReadVariableOp7x_deep_fm_1/linear_1/dense_5/kernel/Read/ReadVariableOp5x_deep_fm_1/linear_1/dense_5/bias/Read/ReadVariableOp<x_deep_fm_1/dense_layer_1/dense_6/kernel/Read/ReadVariableOp:x_deep_fm_1/dense_layer_1/dense_6/bias/Read/ReadVariableOp<x_deep_fm_1/dense_layer_1/dense_7/kernel/Read/ReadVariableOp:x_deep_fm_1/dense_layer_1/dense_7/bias/Read/ReadVariableOp<x_deep_fm_1/dense_layer_1/dense_8/kernel/Read/ReadVariableOp:x_deep_fm_1/dense_layer_1/dense_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOpBAdam/x_deep_fm_1/batch_normalization_1/gamma/m/Read/ReadVariableOpAAdam/x_deep_fm_1/batch_normalization_1/beta/m/Read/ReadVariableOp/Adam/x_deep_fm_1/cin_1/w0/m/Read/ReadVariableOp/Adam/x_deep_fm_1/cin_1/w1/m/Read/ReadVariableOp5Adam/x_deep_fm_1/dense_9/kernel/m/Read/ReadVariableOp3Adam/x_deep_fm_1/dense_9/bias/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_26/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_27/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_28/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_29/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_30/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_31/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_32/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_33/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_34/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_35/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_36/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_37/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_38/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_39/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_40/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_41/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_42/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_43/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_44/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_45/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_46/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_47/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_48/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_49/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_50/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_51/embeddings/m/Read/ReadVariableOp>Adam/x_deep_fm_1/linear_1/dense_5/kernel/m/Read/ReadVariableOp<Adam/x_deep_fm_1/linear_1/dense_5/bias/m/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_6/bias/m/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_7/bias/m/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_8/bias/m/Read/ReadVariableOpBAdam/x_deep_fm_1/batch_normalization_1/gamma/v/Read/ReadVariableOpAAdam/x_deep_fm_1/batch_normalization_1/beta/v/Read/ReadVariableOp/Adam/x_deep_fm_1/cin_1/w0/v/Read/ReadVariableOp/Adam/x_deep_fm_1/cin_1/w1/v/Read/ReadVariableOp5Adam/x_deep_fm_1/dense_9/kernel/v/Read/ReadVariableOp3Adam/x_deep_fm_1/dense_9/bias/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_26/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_27/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_28/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_29/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_30/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_31/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_32/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_33/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_34/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_35/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_36/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_37/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_38/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_39/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_40/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_41/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_42/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_43/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_44/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_45/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_46/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_47/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_48/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_49/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_50/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/embedding_51/embeddings/v/Read/ReadVariableOp>Adam/x_deep_fm_1/linear_1/dense_5/kernel/v/Read/ReadVariableOp<Adam/x_deep_fm_1/linear_1/dense_5/bias/v/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_6/bias/v/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_7/bias/v/Read/ReadVariableOpCAdam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v/Read/ReadVariableOpAAdam/x_deep_fm_1/dense_layer_1/dense_8/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_90167
?,
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'x_deep_fm_1/batch_normalization_1/gamma&x_deep_fm_1/batch_normalization_1/beta-x_deep_fm_1/batch_normalization_1/moving_mean1x_deep_fm_1/batch_normalization_1/moving_variancex_deep_fm_1/cin_1/w0x_deep_fm_1/cin_1/w1x_deep_fm_1/dense_9/kernelx_deep_fm_1/dense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate#x_deep_fm_1/embedding_26/embeddings#x_deep_fm_1/embedding_27/embeddings#x_deep_fm_1/embedding_28/embeddings#x_deep_fm_1/embedding_29/embeddings#x_deep_fm_1/embedding_30/embeddings#x_deep_fm_1/embedding_31/embeddings#x_deep_fm_1/embedding_32/embeddings#x_deep_fm_1/embedding_33/embeddings#x_deep_fm_1/embedding_34/embeddings#x_deep_fm_1/embedding_35/embeddings#x_deep_fm_1/embedding_36/embeddings#x_deep_fm_1/embedding_37/embeddings#x_deep_fm_1/embedding_38/embeddings#x_deep_fm_1/embedding_39/embeddings#x_deep_fm_1/embedding_40/embeddings#x_deep_fm_1/embedding_41/embeddings#x_deep_fm_1/embedding_42/embeddings#x_deep_fm_1/embedding_43/embeddings#x_deep_fm_1/embedding_44/embeddings#x_deep_fm_1/embedding_45/embeddings#x_deep_fm_1/embedding_46/embeddings#x_deep_fm_1/embedding_47/embeddings#x_deep_fm_1/embedding_48/embeddings#x_deep_fm_1/embedding_49/embeddings#x_deep_fm_1/embedding_50/embeddings#x_deep_fm_1/embedding_51/embeddings#x_deep_fm_1/linear_1/dense_5/kernel!x_deep_fm_1/linear_1/dense_5/bias(x_deep_fm_1/dense_layer_1/dense_6/kernel&x_deep_fm_1/dense_layer_1/dense_6/bias(x_deep_fm_1/dense_layer_1/dense_7/kernel&x_deep_fm_1/dense_layer_1/dense_7/bias(x_deep_fm_1/dense_layer_1/dense_8/kernel&x_deep_fm_1/dense_layer_1/dense_8/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_negatives_1.Adam/x_deep_fm_1/batch_normalization_1/gamma/m-Adam/x_deep_fm_1/batch_normalization_1/beta/mAdam/x_deep_fm_1/cin_1/w0/mAdam/x_deep_fm_1/cin_1/w1/m!Adam/x_deep_fm_1/dense_9/kernel/mAdam/x_deep_fm_1/dense_9/bias/m*Adam/x_deep_fm_1/embedding_26/embeddings/m*Adam/x_deep_fm_1/embedding_27/embeddings/m*Adam/x_deep_fm_1/embedding_28/embeddings/m*Adam/x_deep_fm_1/embedding_29/embeddings/m*Adam/x_deep_fm_1/embedding_30/embeddings/m*Adam/x_deep_fm_1/embedding_31/embeddings/m*Adam/x_deep_fm_1/embedding_32/embeddings/m*Adam/x_deep_fm_1/embedding_33/embeddings/m*Adam/x_deep_fm_1/embedding_34/embeddings/m*Adam/x_deep_fm_1/embedding_35/embeddings/m*Adam/x_deep_fm_1/embedding_36/embeddings/m*Adam/x_deep_fm_1/embedding_37/embeddings/m*Adam/x_deep_fm_1/embedding_38/embeddings/m*Adam/x_deep_fm_1/embedding_39/embeddings/m*Adam/x_deep_fm_1/embedding_40/embeddings/m*Adam/x_deep_fm_1/embedding_41/embeddings/m*Adam/x_deep_fm_1/embedding_42/embeddings/m*Adam/x_deep_fm_1/embedding_43/embeddings/m*Adam/x_deep_fm_1/embedding_44/embeddings/m*Adam/x_deep_fm_1/embedding_45/embeddings/m*Adam/x_deep_fm_1/embedding_46/embeddings/m*Adam/x_deep_fm_1/embedding_47/embeddings/m*Adam/x_deep_fm_1/embedding_48/embeddings/m*Adam/x_deep_fm_1/embedding_49/embeddings/m*Adam/x_deep_fm_1/embedding_50/embeddings/m*Adam/x_deep_fm_1/embedding_51/embeddings/m*Adam/x_deep_fm_1/linear_1/dense_5/kernel/m(Adam/x_deep_fm_1/linear_1/dense_5/bias/m/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/m/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/m/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/m.Adam/x_deep_fm_1/batch_normalization_1/gamma/v-Adam/x_deep_fm_1/batch_normalization_1/beta/vAdam/x_deep_fm_1/cin_1/w0/vAdam/x_deep_fm_1/cin_1/w1/v!Adam/x_deep_fm_1/dense_9/kernel/vAdam/x_deep_fm_1/dense_9/bias/v*Adam/x_deep_fm_1/embedding_26/embeddings/v*Adam/x_deep_fm_1/embedding_27/embeddings/v*Adam/x_deep_fm_1/embedding_28/embeddings/v*Adam/x_deep_fm_1/embedding_29/embeddings/v*Adam/x_deep_fm_1/embedding_30/embeddings/v*Adam/x_deep_fm_1/embedding_31/embeddings/v*Adam/x_deep_fm_1/embedding_32/embeddings/v*Adam/x_deep_fm_1/embedding_33/embeddings/v*Adam/x_deep_fm_1/embedding_34/embeddings/v*Adam/x_deep_fm_1/embedding_35/embeddings/v*Adam/x_deep_fm_1/embedding_36/embeddings/v*Adam/x_deep_fm_1/embedding_37/embeddings/v*Adam/x_deep_fm_1/embedding_38/embeddings/v*Adam/x_deep_fm_1/embedding_39/embeddings/v*Adam/x_deep_fm_1/embedding_40/embeddings/v*Adam/x_deep_fm_1/embedding_41/embeddings/v*Adam/x_deep_fm_1/embedding_42/embeddings/v*Adam/x_deep_fm_1/embedding_43/embeddings/v*Adam/x_deep_fm_1/embedding_44/embeddings/v*Adam/x_deep_fm_1/embedding_45/embeddings/v*Adam/x_deep_fm_1/embedding_46/embeddings/v*Adam/x_deep_fm_1/embedding_47/embeddings/v*Adam/x_deep_fm_1/embedding_48/embeddings/v*Adam/x_deep_fm_1/embedding_49/embeddings/v*Adam/x_deep_fm_1/embedding_50/embeddings/v*Adam/x_deep_fm_1/embedding_51/embeddings/v*Adam/x_deep_fm_1/linear_1/dense_5/kernel/v(Adam/x_deep_fm_1/linear_1/dense_5/bias/v/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/v/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/v/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_90582??/
?

?
G__inference_embedding_32_layer_call_and_return_conditional_losses_85483

inputs)
embedding_lookup_85477:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85477Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85477*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85477*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_89297S
<x_deep_fm_1_cin_1_w1_regularizer_abs_readvariableop_resource:? 
identity??3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp<x_deep_fm_1_cin_1_w1_regularizer_abs_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp<x_deep_fm_1_cin_1_w1_regularizer_abs_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1t
IdentityIdentity*x_deep_fm_1/cin_1/w1/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp
?

?
G__inference_embedding_29_layer_call_and_return_conditional_losses_85429

inputs)
embedding_lookup_85423:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85423Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85423*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85423*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_linear_1_layer_call_and_return_conditional_losses_86534

inputs8
&dense_5_matmul_readvariableop_resource:'5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_49_layer_call_and_return_conditional_losses_85789

inputs)
embedding_lookup_85783:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85783Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85783*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85783*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_cin_1_layer_call_fn_89040

inputs
unknown:?  
	unknown_0:? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_862972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_51_layer_call_and_return_conditional_losses_89739

inputs)
embedding_lookup_89733:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89733Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89733*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89733*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_49_layer_call_fn_89695

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_857892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_47_layer_call_and_return_conditional_losses_85753

inputs(
embedding_lookup_85747:

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85747Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85747*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85747*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_40_layer_call_and_return_conditional_losses_85627

inputs)
embedding_lookup_85621:	?

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85621Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85621*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85621*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_88804

inputs
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_851832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_35_layer_call_and_return_conditional_losses_85537

inputs)
embedding_lookup_85531:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85531Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85531*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85531*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_46_layer_call_and_return_conditional_losses_89654

inputs)
embedding_lookup_89648:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89648Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89648*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89648*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_42_layer_call_and_return_conditional_losses_89586

inputs(
embedding_lookup_89580:

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89580Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89580*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89580*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_36_layer_call_and_return_conditional_losses_89484

inputs)
embedding_lookup_89478:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89478Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89478*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89478*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_45_layer_call_fn_89627

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_857172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_49_layer_call_and_return_conditional_losses_89705

inputs)
embedding_lookup_89699:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89699Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89699*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89699*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_38_layer_call_fn_89508

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_855912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
-__inference_dense_layer_1_layer_call_fn_88926

inputs
unknown:	?@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_859662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
-__inference_dense_layer_1_layer_call_fn_88943

inputs
unknown:	?@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_861712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_32_layer_call_and_return_conditional_losses_89416

inputs)
embedding_lookup_89410:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89410Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89410*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89410*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_30_layer_call_and_return_conditional_losses_89382

inputs(
embedding_lookup_89376:*
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89376Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89376*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89376*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_45_layer_call_and_return_conditional_losses_85717

inputs(
embedding_lookup_85711:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85711Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85711*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85711*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_86903

inputs)
batch_normalization_1_86635:')
batch_normalization_1_86637:')
batch_normalization_1_86639:')
batch_normalization_1_86641:' 
linear_1_86652:'
linear_1_86654:$
embedding_26_86661:h%
embedding_27_86668:	?%
embedding_28_86675:	?%
embedding_29_86682:	?$
embedding_30_86689:*$
embedding_31_86696:	%
embedding_32_86703:	?$
embedding_33_86710:C$
embedding_34_86717:%
embedding_35_86724:	?%
embedding_36_86731:	?%
embedding_37_86738:	?%
embedding_38_86745:	?$
embedding_39_86752:%
embedding_40_86759:	?
%
embedding_41_86766:	?$
embedding_42_86773:
%
embedding_43_86780:	?%
embedding_44_86787:	?$
embedding_45_86794:%
embedding_46_86801:	?$
embedding_47_86808:
$
embedding_48_86815:%
embedding_49_86822:	?$
embedding_50_86829:'%
embedding_51_86836:	?"
cin_1_86842:? "
cin_1_86844:? &
dense_layer_1_86851:	?@!
dense_layer_1_86853:@%
dense_layer_1_86855:@@!
dense_layer_1_86857:@%
dense_layer_1_86859:@!
dense_layer_1_86861:
dense_9_86866:@
dense_9_86868:
identity??-batch_normalization_1/StatefulPartitionedCall?cin_1/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?$embedding_36/StatefulPartitionedCall?$embedding_37/StatefulPartitionedCall?$embedding_38/StatefulPartitionedCall?$embedding_39/StatefulPartitionedCall?$embedding_40/StatefulPartitionedCall?$embedding_41/StatefulPartitionedCall?$embedding_42/StatefulPartitionedCall?$embedding_43/StatefulPartitionedCall?$embedding_44/StatefulPartitionedCall?$embedding_45/StatefulPartitionedCall?$embedding_46/StatefulPartitionedCall?$embedding_47/StatefulPartitionedCall?$embedding_48/StatefulPartitionedCall?$embedding_49/StatefulPartitionedCall?$embedding_50/StatefulPartitionedCall?$embedding_51/StatefulPartitionedCall? linear_1/StatefulPartitionedCall?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_86635batch_normalization_1_86637batch_normalization_1_86639batch_normalization_1_86641*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_852432/
-batch_normalization_1/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
 linear_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0linear_1_86652linear_1_86654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_865342"
 linear_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_86661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_853752&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_86668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_853932&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_86675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_854112&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_86682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_854292&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_86689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_854472&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_86696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_854652&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_86703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_854832&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_86710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_855012&
$embedding_33/StatefulPartitionedCall?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_86717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_855192&
$embedding_34/StatefulPartitionedCall?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_86724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_855372&
$embedding_35/StatefulPartitionedCall?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_86731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_855552&
$embedding_36/StatefulPartitionedCall?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_86738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_855732&
$embedding_37/StatefulPartitionedCall?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_86745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_855912&
$embedding_38/StatefulPartitionedCall?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_86752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_856092&
$embedding_39/StatefulPartitionedCall?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_86759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_856272&
$embedding_40/StatefulPartitionedCall?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_86766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_856452&
$embedding_41/StatefulPartitionedCall?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_86773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_856632&
$embedding_42/StatefulPartitionedCall?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_86780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_856812&
$embedding_43/StatefulPartitionedCall?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_86787*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_856992&
$embedding_44/StatefulPartitionedCall?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_86794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_857172&
$embedding_45/StatefulPartitionedCall?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_86801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_857352&
$embedding_46/StatefulPartitionedCall?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_86808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_857532&
$embedding_47/StatefulPartitionedCall?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_86815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_857712&
$embedding_48/StatefulPartitionedCall?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_86822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_857892&
$embedding_49/StatefulPartitionedCall?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_86829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_858072&
$embedding_50/StatefulPartitionedCall?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_86836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_858252&
$embedding_51/StatefulPartitionedCall?

packedPack-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transpose?
cin_1/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0cin_1_86842cin_1_86844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_862972
cin_1/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_layer_1_86851dense_layer_1_86853dense_layer_1_86855dense_layer_1_86857dense_layer_1_86859dense_layer_1_86861*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_861712'
%dense_layer_1/StatefulPartitionedCall?
addAddV2)linear_1/StatefulPartitionedCall:output:0&cin_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add?
add_1AddV2add:z:0.dense_layer_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0dense_9_86866dense_9_86868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_859922!
dense_9/StatefulPartitionedCally
SigmoidSigmoid(dense_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_86842*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_86842*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_86844*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_86844*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^cin_1/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall!^linear_1/StatefulPartitionedCall4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
cin_1/StatefulPartitionedCallcin_1/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall2L
$embedding_36/StatefulPartitionedCall$embedding_36/StatefulPartitionedCall2L
$embedding_37/StatefulPartitionedCall$embedding_37/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2L
$embedding_39/StatefulPartitionedCall$embedding_39/StatefulPartitionedCall2L
$embedding_40/StatefulPartitionedCall$embedding_40/StatefulPartitionedCall2L
$embedding_41/StatefulPartitionedCall$embedding_41/StatefulPartitionedCall2L
$embedding_42/StatefulPartitionedCall$embedding_42/StatefulPartitionedCall2L
$embedding_43/StatefulPartitionedCall$embedding_43/StatefulPartitionedCall2L
$embedding_44/StatefulPartitionedCall$embedding_44/StatefulPartitionedCall2L
$embedding_45/StatefulPartitionedCall$embedding_45/StatefulPartitionedCall2L
$embedding_46/StatefulPartitionedCall$embedding_46/StatefulPartitionedCall2L
$embedding_47/StatefulPartitionedCall$embedding_47/StatefulPartitionedCall2L
$embedding_48/StatefulPartitionedCall$embedding_48/StatefulPartitionedCall2L
$embedding_49/StatefulPartitionedCall$embedding_49/StatefulPartitionedCall2L
$embedding_50/StatefulPartitionedCall$embedding_50/StatefulPartitionedCall2L
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall2D
 linear_1/StatefulPartitionedCall linear_1/StatefulPartitionedCall2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?

#__inference_signature_wrapper_87748
input_1
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
	unknown_3:'
	unknown_4:
	unknown_5:h
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:*

unknown_10:	

unknown_11:	?

unknown_12:C

unknown_13:

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?


unknown_20:	?

unknown_21:


unknown_22:	?

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:


unknown_27:

unknown_28:	?

unknown_29:'

unknown_30:	?!

unknown_31:? !

unknown_32:? 

unknown_33:	?@

unknown_34:@

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:

unknown_39:@

unknown_40:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_851592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
?
?
,__inference_embedding_39_layer_call_fn_89525

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_856092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_38_layer_call_and_return_conditional_losses_89518

inputs)
embedding_lookup_89512:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89512Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89512*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89512*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_86030

inputs)
batch_normalization_1_85328:')
batch_normalization_1_85330:')
batch_normalization_1_85332:')
batch_normalization_1_85334:' 
linear_1_85356:'
linear_1_85358:$
embedding_26_85376:h%
embedding_27_85394:	?%
embedding_28_85412:	?%
embedding_29_85430:	?$
embedding_30_85448:*$
embedding_31_85466:	%
embedding_32_85484:	?$
embedding_33_85502:C$
embedding_34_85520:%
embedding_35_85538:	?%
embedding_36_85556:	?%
embedding_37_85574:	?%
embedding_38_85592:	?$
embedding_39_85610:%
embedding_40_85628:	?
%
embedding_41_85646:	?$
embedding_42_85664:
%
embedding_43_85682:	?%
embedding_44_85700:	?$
embedding_45_85718:%
embedding_46_85736:	?$
embedding_47_85754:
$
embedding_48_85772:%
embedding_49_85790:	?$
embedding_50_85808:'%
embedding_51_85826:	?"
cin_1_85932:? "
cin_1_85934:? &
dense_layer_1_85967:	?@!
dense_layer_1_85969:@%
dense_layer_1_85971:@@!
dense_layer_1_85973:@%
dense_layer_1_85975:@!
dense_layer_1_85977:
dense_9_85993:@
dense_9_85995:
identity??-batch_normalization_1/StatefulPartitionedCall?cin_1/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?$embedding_36/StatefulPartitionedCall?$embedding_37/StatefulPartitionedCall?$embedding_38/StatefulPartitionedCall?$embedding_39/StatefulPartitionedCall?$embedding_40/StatefulPartitionedCall?$embedding_41/StatefulPartitionedCall?$embedding_42/StatefulPartitionedCall?$embedding_43/StatefulPartitionedCall?$embedding_44/StatefulPartitionedCall?$embedding_45/StatefulPartitionedCall?$embedding_46/StatefulPartitionedCall?$embedding_47/StatefulPartitionedCall?$embedding_48/StatefulPartitionedCall?$embedding_49/StatefulPartitionedCall?$embedding_50/StatefulPartitionedCall?$embedding_51/StatefulPartitionedCall? linear_1/StatefulPartitionedCall?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_85328batch_normalization_1_85330batch_normalization_1_85332batch_normalization_1_85334*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_851832/
-batch_normalization_1/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
 linear_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0linear_1_85356linear_1_85358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_853552"
 linear_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_85376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_853752&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_85394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_853932&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_85412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_854112&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_85430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_854292&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_85448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_854472&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_85466*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_854652&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_85484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_854832&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_85502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_855012&
$embedding_33/StatefulPartitionedCall?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_85520*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_855192&
$embedding_34/StatefulPartitionedCall?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_85538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_855372&
$embedding_35/StatefulPartitionedCall?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_85556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_855552&
$embedding_36/StatefulPartitionedCall?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_85574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_855732&
$embedding_37/StatefulPartitionedCall?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_85592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_855912&
$embedding_38/StatefulPartitionedCall?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_85610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_856092&
$embedding_39/StatefulPartitionedCall?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_85628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_856272&
$embedding_40/StatefulPartitionedCall?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_85646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_856452&
$embedding_41/StatefulPartitionedCall?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_85664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_856632&
$embedding_42/StatefulPartitionedCall?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_85682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_856812&
$embedding_43/StatefulPartitionedCall?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_85700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_856992&
$embedding_44/StatefulPartitionedCall?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_85718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_857172&
$embedding_45/StatefulPartitionedCall?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_85736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_857352&
$embedding_46/StatefulPartitionedCall?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_85754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_857532&
$embedding_47/StatefulPartitionedCall?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_85772*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_857712&
$embedding_48/StatefulPartitionedCall?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_85790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_857892&
$embedding_49/StatefulPartitionedCall?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_85808*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_858072&
$embedding_50/StatefulPartitionedCall?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_85826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_858252&
$embedding_51/StatefulPartitionedCall?

packedPack-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transpose?
cin_1/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0cin_1_85932cin_1_85934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_859312
cin_1/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_layer_1_85967dense_layer_1_85969dense_layer_1_85971dense_layer_1_85973dense_layer_1_85975dense_layer_1_85977*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_859662'
%dense_layer_1/StatefulPartitionedCall?
addAddV2)linear_1/StatefulPartitionedCall:output:0&cin_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add?
add_1AddV2add:z:0.dense_layer_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0dense_9_85993dense_9_85995*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_859922!
dense_9/StatefulPartitionedCally
SigmoidSigmoid(dense_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_85932*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_85932*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_85934*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_85934*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^cin_1/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall!^linear_1/StatefulPartitionedCall4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
cin_1/StatefulPartitionedCallcin_1/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall2L
$embedding_36/StatefulPartitionedCall$embedding_36/StatefulPartitionedCall2L
$embedding_37/StatefulPartitionedCall$embedding_37/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2L
$embedding_39/StatefulPartitionedCall$embedding_39/StatefulPartitionedCall2L
$embedding_40/StatefulPartitionedCall$embedding_40/StatefulPartitionedCall2L
$embedding_41/StatefulPartitionedCall$embedding_41/StatefulPartitionedCall2L
$embedding_42/StatefulPartitionedCall$embedding_42/StatefulPartitionedCall2L
$embedding_43/StatefulPartitionedCall$embedding_43/StatefulPartitionedCall2L
$embedding_44/StatefulPartitionedCall$embedding_44/StatefulPartitionedCall2L
$embedding_45/StatefulPartitionedCall$embedding_45/StatefulPartitionedCall2L
$embedding_46/StatefulPartitionedCall$embedding_46/StatefulPartitionedCall2L
$embedding_47/StatefulPartitionedCall$embedding_47/StatefulPartitionedCall2L
$embedding_48/StatefulPartitionedCall$embedding_48/StatefulPartitionedCall2L
$embedding_49/StatefulPartitionedCall$embedding_49/StatefulPartitionedCall2L
$embedding_50/StatefulPartitionedCall$embedding_50/StatefulPartitionedCall2L
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall2D
 linear_1/StatefulPartitionedCall linear_1/StatefulPartitionedCall2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_50_layer_call_and_return_conditional_losses_85807

inputs(
embedding_lookup_85801:'
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85801Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85801*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85801*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_28_layer_call_and_return_conditional_losses_89348

inputs)
embedding_lookup_89342:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89342Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89342*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89342*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
?
@__inference_cin_1_layer_call_and_return_conditional_losses_89238

inputsB
+conv1d_expanddims_1_readvariableop_resource:? D
-conv1d_1_expanddims_1_readvariableop_resource:? 
identity??"conv1d/ExpandDims_1/ReadVariableOp?$conv1d_1/ExpandDims_1/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
splitq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2	
split_1?
MatMul/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/a?
MatMul/bPacksplit_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/b?
MatMulBatchMatMulV2MatMul/a:output:0MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
MatMuls
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
Reshape/shape}
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapeu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposey
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimstranspose:y:0conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeezey
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transposeconv1d/Squeeze:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1q
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2	
split_2?

MatMul_1/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul_1/a?

MatMul_1/bPacksplit_2:output:0split_2:output:1split_2:output:2split_2:output:3split_2:output:4split_2:output:5split_2:output:6split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2

MatMul_1/b?
MatMul_1BatchMatMulV2MatMul_1/a:output:0MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2

MatMul_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
Reshape_1/shape?
	Reshape_1ReshapeMatMul_1:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
	Reshape_1y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_1:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2}
conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_1/ExpandDims/dim?
conv1d_1/ExpandDims
ExpandDimstranspose_2:y:0 conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_1/ExpandDims?
$conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02&
$conv1d_1/ExpandDims_1/ReadVariableOpx
conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_1/ExpandDims_1/dim?
conv1d_1/ExpandDims_1
ExpandDims,conv1d_1/ExpandDims_1/ReadVariableOp:value:0"conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d_1/ExpandDims_1?
conv1d_1Conv2Dconv1d_1/ExpandDims:output:0conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2

conv1d_1?
conv1d_1/SqueezeSqueezeconv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/Squeezey
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm?
transpose_3	Transposeconv1d_1/Squeeze:output:0transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2transpose_1:y:0transpose_3:y:0concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicest
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1g
IdentityIdentitySum:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp%^conv1d_1/ExpandDims_1/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2L
$conv1d_1/ExpandDims_1/ReadVariableOp$conv1d_1/ExpandDims_1/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_9_layer_call_fn_89247

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_859922
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
%__inference_cin_1_layer_call_fn_89031

inputs
unknown:?  
	unknown_0:? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_859312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_48_layer_call_fn_89678

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_857712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

+__inference_x_deep_fm_1_layer_call_fn_87926

inputs
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
	unknown_3:'
	unknown_4:
	unknown_5:h
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:*

unknown_10:	

unknown_11:	?

unknown_12:C

unknown_13:

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?


unknown_20:	?

unknown_21:


unknown_22:	?

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:


unknown_27:

unknown_28:	?

unknown_29:'

unknown_30:	?!

unknown_31:? !

unknown_32:? 

unknown_33:	?@

unknown_34:@

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:

unknown_39:@

unknown_40:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_869032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?q
?
@__inference_cin_1_layer_call_and_return_conditional_losses_86297

inputsB
+conv1d_expanddims_1_readvariableop_resource:? D
-conv1d_1_expanddims_1_readvariableop_resource:? 
identity??"conv1d/ExpandDims_1/ReadVariableOp?$conv1d_1/ExpandDims_1/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
splitq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2	
split_1?
MatMul/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/a?
MatMul/bPacksplit_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/b?
MatMulBatchMatMulV2MatMul/a:output:0MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
MatMuls
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
Reshape/shape}
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapeu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposey
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimstranspose:y:0conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeezey
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transposeconv1d/Squeeze:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1q
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2	
split_2?

MatMul_1/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul_1/a?

MatMul_1/bPacksplit_2:output:0split_2:output:1split_2:output:2split_2:output:3split_2:output:4split_2:output:5split_2:output:6split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2

MatMul_1/b?
MatMul_1BatchMatMulV2MatMul_1/a:output:0MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2

MatMul_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
Reshape_1/shape?
	Reshape_1ReshapeMatMul_1:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
	Reshape_1y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_1:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2}
conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_1/ExpandDims/dim?
conv1d_1/ExpandDims
ExpandDimstranspose_2:y:0 conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_1/ExpandDims?
$conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02&
$conv1d_1/ExpandDims_1/ReadVariableOpx
conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_1/ExpandDims_1/dim?
conv1d_1/ExpandDims_1
ExpandDims,conv1d_1/ExpandDims_1/ReadVariableOp:value:0"conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d_1/ExpandDims_1?
conv1d_1Conv2Dconv1d_1/ExpandDims:output:0conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2

conv1d_1?
conv1d_1/SqueezeSqueezeconv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/Squeezey
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm?
transpose_3	Transposeconv1d_1/Squeeze:output:0transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2transpose_1:y:0transpose_3:y:0concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicest
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1g
IdentityIdentitySum:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp%^conv1d_1/ExpandDims_1/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2L
$conv1d_1/ExpandDims_1/ReadVariableOp$conv1d_1/ExpandDims_1/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_linear_1_layer_call_and_return_conditional_losses_85355

inputs8
&dense_5_matmul_readvariableop_resource:'5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_26_layer_call_and_return_conditional_losses_89314

inputs(
embedding_lookup_89308:h
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89308Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89308*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89308*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_86171

inputs9
&dense_6_matmul_readvariableop_resource:	?@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?

+__inference_x_deep_fm_1_layer_call_fn_87079
input_1
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
	unknown_3:'
	unknown_4:
	unknown_5:h
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:*

unknown_10:	

unknown_11:	?

unknown_12:C

unknown_13:

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?


unknown_20:	?

unknown_21:


unknown_22:	?

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:


unknown_27:

unknown_28:	?

unknown_29:'

unknown_30:	?!

unknown_31:? !

unknown_32:? 

unknown_33:	?@

unknown_34:@

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:

unknown_39:@

unknown_40:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_869032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
?)
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88871

inputs5
'assignmovingavg_readvariableop_resource:'7
)assignmovingavg_1_readvariableop_resource:'*
cast_readvariableop_resource:',
cast_1_readvariableop_resource:'
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:'2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????'2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:'*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:'*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:'2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:'2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:'2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:'2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_37_layer_call_and_return_conditional_losses_85573

inputs)
embedding_lookup_85567:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85567Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85567*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85567*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_27_layer_call_fn_89321

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_853932
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88968

inputs9
&dense_6_matmul_readvariableop_resource:	?@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
dropout_1/IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_1/Identity?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_1/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_43_layer_call_fn_89593

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_856812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_34_layer_call_and_return_conditional_losses_85519

inputs(
embedding_lookup_85513:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85513Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85513*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85513*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_33_layer_call_and_return_conditional_losses_89433

inputs(
embedding_lookup_89427:C
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89427Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89427*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89427*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_30_layer_call_fn_89372

inputs
unknown:*
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_854472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
Д
?J
__inference__traced_save_90167
file_prefixF
Bsavev2_x_deep_fm_1_batch_normalization_1_gamma_read_readvariableopE
Asavev2_x_deep_fm_1_batch_normalization_1_beta_read_readvariableopL
Hsavev2_x_deep_fm_1_batch_normalization_1_moving_mean_read_readvariableopP
Lsavev2_x_deep_fm_1_batch_normalization_1_moving_variance_read_readvariableop3
/savev2_x_deep_fm_1_cin_1_w0_read_readvariableop3
/savev2_x_deep_fm_1_cin_1_w1_read_readvariableop9
5savev2_x_deep_fm_1_dense_9_kernel_read_readvariableop7
3savev2_x_deep_fm_1_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopB
>savev2_x_deep_fm_1_embedding_26_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_27_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_28_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_29_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_30_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_31_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_32_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_33_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_34_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_35_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_36_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_37_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_38_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_39_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_40_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_41_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_42_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_43_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_44_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_45_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_46_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_47_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_48_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_49_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_50_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_embedding_51_embeddings_read_readvariableopB
>savev2_x_deep_fm_1_linear_1_dense_5_kernel_read_readvariableop@
<savev2_x_deep_fm_1_linear_1_dense_5_bias_read_readvariableopG
Csavev2_x_deep_fm_1_dense_layer_1_dense_6_kernel_read_readvariableopE
Asavev2_x_deep_fm_1_dense_layer_1_dense_6_bias_read_readvariableopG
Csavev2_x_deep_fm_1_dense_layer_1_dense_7_kernel_read_readvariableopE
Asavev2_x_deep_fm_1_dense_layer_1_dense_7_bias_read_readvariableopG
Csavev2_x_deep_fm_1_dense_layer_1_dense_8_kernel_read_readvariableopE
Asavev2_x_deep_fm_1_dense_layer_1_dense_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableopM
Isavev2_adam_x_deep_fm_1_batch_normalization_1_gamma_m_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_batch_normalization_1_beta_m_read_readvariableop:
6savev2_adam_x_deep_fm_1_cin_1_w0_m_read_readvariableop:
6savev2_adam_x_deep_fm_1_cin_1_w1_m_read_readvariableop@
<savev2_adam_x_deep_fm_1_dense_9_kernel_m_read_readvariableop>
:savev2_adam_x_deep_fm_1_dense_9_bias_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_26_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_27_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_28_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_29_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_30_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_31_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_32_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_33_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_34_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_35_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_36_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_37_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_38_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_39_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_40_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_41_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_42_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_43_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_44_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_45_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_46_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_47_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_48_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_49_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_50_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_51_embeddings_m_read_readvariableopI
Esavev2_adam_x_deep_fm_1_linear_1_dense_5_kernel_m_read_readvariableopG
Csavev2_adam_x_deep_fm_1_linear_1_dense_5_bias_m_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_m_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_m_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_m_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_m_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_m_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_m_read_readvariableopM
Isavev2_adam_x_deep_fm_1_batch_normalization_1_gamma_v_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_batch_normalization_1_beta_v_read_readvariableop:
6savev2_adam_x_deep_fm_1_cin_1_w0_v_read_readvariableop:
6savev2_adam_x_deep_fm_1_cin_1_w1_v_read_readvariableop@
<savev2_adam_x_deep_fm_1_dense_9_kernel_v_read_readvariableop>
:savev2_adam_x_deep_fm_1_dense_9_bias_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_26_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_27_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_28_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_29_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_30_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_31_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_32_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_33_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_34_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_35_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_36_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_37_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_38_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_39_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_40_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_41_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_42_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_43_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_44_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_45_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_46_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_47_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_48_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_49_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_50_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_embedding_51_embeddings_v_read_readvariableopI
Esavev2_adam_x_deep_fm_1_linear_1_dense_5_kernel_v_read_readvariableopG
Csavev2_adam_x_deep_fm_1_linear_1_dense_5_bias_v_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_v_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_v_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_v_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_v_read_readvariableopN
Jsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_v_read_readvariableopL
Hsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename??
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?>
value?>B?>?B#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'cin_layer/w0/.ATTRIBUTES/VARIABLE_VALUEB'cin_layer/w1/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEout_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEout_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?G
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_x_deep_fm_1_batch_normalization_1_gamma_read_readvariableopAsavev2_x_deep_fm_1_batch_normalization_1_beta_read_readvariableopHsavev2_x_deep_fm_1_batch_normalization_1_moving_mean_read_readvariableopLsavev2_x_deep_fm_1_batch_normalization_1_moving_variance_read_readvariableop/savev2_x_deep_fm_1_cin_1_w0_read_readvariableop/savev2_x_deep_fm_1_cin_1_w1_read_readvariableop5savev2_x_deep_fm_1_dense_9_kernel_read_readvariableop3savev2_x_deep_fm_1_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop>savev2_x_deep_fm_1_embedding_26_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_27_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_28_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_29_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_30_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_31_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_32_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_33_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_34_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_35_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_36_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_37_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_38_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_39_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_40_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_41_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_42_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_43_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_44_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_45_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_46_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_47_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_48_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_49_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_50_embeddings_read_readvariableop>savev2_x_deep_fm_1_embedding_51_embeddings_read_readvariableop>savev2_x_deep_fm_1_linear_1_dense_5_kernel_read_readvariableop<savev2_x_deep_fm_1_linear_1_dense_5_bias_read_readvariableopCsavev2_x_deep_fm_1_dense_layer_1_dense_6_kernel_read_readvariableopAsavev2_x_deep_fm_1_dense_layer_1_dense_6_bias_read_readvariableopCsavev2_x_deep_fm_1_dense_layer_1_dense_7_kernel_read_readvariableopAsavev2_x_deep_fm_1_dense_layer_1_dense_7_bias_read_readvariableopCsavev2_x_deep_fm_1_dense_layer_1_dense_8_kernel_read_readvariableopAsavev2_x_deep_fm_1_dense_layer_1_dense_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableopIsavev2_adam_x_deep_fm_1_batch_normalization_1_gamma_m_read_readvariableopHsavev2_adam_x_deep_fm_1_batch_normalization_1_beta_m_read_readvariableop6savev2_adam_x_deep_fm_1_cin_1_w0_m_read_readvariableop6savev2_adam_x_deep_fm_1_cin_1_w1_m_read_readvariableop<savev2_adam_x_deep_fm_1_dense_9_kernel_m_read_readvariableop:savev2_adam_x_deep_fm_1_dense_9_bias_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_26_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_27_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_28_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_29_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_30_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_31_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_32_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_33_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_34_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_35_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_36_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_37_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_38_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_39_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_40_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_41_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_42_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_43_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_44_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_45_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_46_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_47_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_48_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_49_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_50_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_51_embeddings_m_read_readvariableopEsavev2_adam_x_deep_fm_1_linear_1_dense_5_kernel_m_read_readvariableopCsavev2_adam_x_deep_fm_1_linear_1_dense_5_bias_m_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_m_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_m_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_m_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_m_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_m_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_m_read_readvariableopIsavev2_adam_x_deep_fm_1_batch_normalization_1_gamma_v_read_readvariableopHsavev2_adam_x_deep_fm_1_batch_normalization_1_beta_v_read_readvariableop6savev2_adam_x_deep_fm_1_cin_1_w0_v_read_readvariableop6savev2_adam_x_deep_fm_1_cin_1_w1_v_read_readvariableop<savev2_adam_x_deep_fm_1_dense_9_kernel_v_read_readvariableop:savev2_adam_x_deep_fm_1_dense_9_bias_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_26_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_27_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_28_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_29_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_30_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_31_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_32_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_33_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_34_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_35_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_36_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_37_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_38_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_39_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_40_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_41_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_42_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_43_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_44_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_45_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_46_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_47_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_48_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_49_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_50_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_embedding_51_embeddings_v_read_readvariableopEsavev2_adam_x_deep_fm_1_linear_1_dense_5_kernel_v_read_readvariableopCsavev2_adam_x_deep_fm_1_linear_1_dense_5_bias_v_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_v_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_v_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_v_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_v_read_readvariableopJsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_v_read_readvariableopHsavev2_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?

_input_shapes?	
?	: :':':':':? :? :@:: : : : : :h:	?:	?:	?:*:	:	?:C::	?:	?:	?:	?::	?
:	?:
:	?:	?::	?:
::	?:':	?:'::	?@:@:@@:@:@:: : :?:?:?:?:::':':? :? :@::h:	?:	?:	?:*:	:	?:C::	?:	?:	?:	?::	?
:	?:
:	?:	?::	?:
::	?:':	?:'::	?@:@:@@:@:@::':':? :? :@::h:	?:	?:	?:*:	:	?:C::	?:	?:	?:	?::	?
:	?:
:	?:	?::	?:
::	?:':	?:'::	?@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:': 

_output_shapes
:': 

_output_shapes
:': 

_output_shapes
:':)%
#
_output_shapes
:? :)%
#
_output_shapes
:? :$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:h:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:$ 

_output_shapes

:*:$ 

_output_shapes

:	:%!

_output_shapes
:	?:$ 

_output_shapes

:C:$ 

_output_shapes

::%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:%!

_output_shapes
:	?:$ 

_output_shapes

::%!

_output_shapes
:	?
:%!

_output_shapes
:	?:$ 

_output_shapes

:
:%!

_output_shapes
:	?:% !

_output_shapes
:	?:$! 

_output_shapes

::%"!

_output_shapes
:	?:$# 

_output_shapes

:
:$$ 

_output_shapes

::%%!

_output_shapes
:	?:$& 

_output_shapes

:':%'!

_output_shapes
:	?:$( 

_output_shapes

:': )

_output_shapes
::%*!

_output_shapes
:	?@: +

_output_shapes
:@:$, 

_output_shapes

:@@: -

_output_shapes
:@:$. 

_output_shapes

:@: /

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :!2

_output_shapes	
:?:!3

_output_shapes	
:?:!4

_output_shapes	
:?:!5

_output_shapes	
:?: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:': 9

_output_shapes
:':):%
#
_output_shapes
:? :);%
#
_output_shapes
:? :$< 

_output_shapes

:@: =

_output_shapes
::$> 

_output_shapes

:h:%?!

_output_shapes
:	?:%@!

_output_shapes
:	?:%A!

_output_shapes
:	?:$B 

_output_shapes

:*:$C 

_output_shapes

:	:%D!

_output_shapes
:	?:$E 

_output_shapes

:C:$F 

_output_shapes

::%G!

_output_shapes
:	?:%H!

_output_shapes
:	?:%I!

_output_shapes
:	?:%J!

_output_shapes
:	?:$K 

_output_shapes

::%L!

_output_shapes
:	?
:%M!

_output_shapes
:	?:$N 

_output_shapes

:
:%O!

_output_shapes
:	?:%P!

_output_shapes
:	?:$Q 

_output_shapes

::%R!

_output_shapes
:	?:$S 

_output_shapes

:
:$T 

_output_shapes

::%U!

_output_shapes
:	?:$V 

_output_shapes

:':%W!

_output_shapes
:	?:$X 

_output_shapes

:': Y

_output_shapes
::%Z!

_output_shapes
:	?@: [

_output_shapes
:@:$\ 

_output_shapes

:@@: ]

_output_shapes
:@:$^ 

_output_shapes

:@: _

_output_shapes
:: `

_output_shapes
:': a

_output_shapes
:':)b%
#
_output_shapes
:? :)c%
#
_output_shapes
:? :$d 

_output_shapes

:@: e

_output_shapes
::$f 

_output_shapes

:h:%g!

_output_shapes
:	?:%h!

_output_shapes
:	?:%i!

_output_shapes
:	?:$j 

_output_shapes

:*:$k 

_output_shapes

:	:%l!

_output_shapes
:	?:$m 

_output_shapes

:C:$n 

_output_shapes

::%o!

_output_shapes
:	?:%p!

_output_shapes
:	?:%q!

_output_shapes
:	?:%r!

_output_shapes
:	?:$s 

_output_shapes

::%t!

_output_shapes
:	?
:%u!

_output_shapes
:	?:$v 

_output_shapes

:
:%w!

_output_shapes
:	?:%x!

_output_shapes
:	?:$y 

_output_shapes

::%z!

_output_shapes
:	?:${ 

_output_shapes

:
:$| 

_output_shapes

::%}!

_output_shapes
:	?:$~ 

_output_shapes

:':%!

_output_shapes
:	?:%? 

_output_shapes

:':!?

_output_shapes
::&?!

_output_shapes
:	?@:!?

_output_shapes
:@:%? 

_output_shapes

:@@:!?

_output_shapes
:@:%? 

_output_shapes

:@:!?

_output_shapes
::?

_output_shapes
: 
?

?
G__inference_embedding_45_layer_call_and_return_conditional_losses_89637

inputs(
embedding_lookup_89631:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89631Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89631*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89631*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_27_layer_call_and_return_conditional_losses_85393

inputs)
embedding_lookup_85387:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85387Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85387*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85387*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_50_layer_call_and_return_conditional_losses_89722

inputs(
embedding_lookup_89716:'
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89716Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89716*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89716*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_linear_1_layer_call_fn_88889

inputs
unknown:'
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_865342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_31_layer_call_and_return_conditional_losses_85465

inputs(
embedding_lookup_85459:	
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85459Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85459*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85459*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_88817

inputs
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_852432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
,__inference_embedding_29_layer_call_fn_89355

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_854292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_26_layer_call_and_return_conditional_losses_85375

inputs(
embedding_lookup_85369:h
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85369Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85369*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85369*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_44_layer_call_fn_89610

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_856992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_46_layer_call_and_return_conditional_losses_85735

inputs)
embedding_lookup_85729:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85729Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85729*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85729*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87621
input_1)
batch_normalization_1_87353:')
batch_normalization_1_87355:')
batch_normalization_1_87357:')
batch_normalization_1_87359:' 
linear_1_87370:'
linear_1_87372:$
embedding_26_87379:h%
embedding_27_87386:	?%
embedding_28_87393:	?%
embedding_29_87400:	?$
embedding_30_87407:*$
embedding_31_87414:	%
embedding_32_87421:	?$
embedding_33_87428:C$
embedding_34_87435:%
embedding_35_87442:	?%
embedding_36_87449:	?%
embedding_37_87456:	?%
embedding_38_87463:	?$
embedding_39_87470:%
embedding_40_87477:	?
%
embedding_41_87484:	?$
embedding_42_87491:
%
embedding_43_87498:	?%
embedding_44_87505:	?$
embedding_45_87512:%
embedding_46_87519:	?$
embedding_47_87526:
$
embedding_48_87533:%
embedding_49_87540:	?$
embedding_50_87547:'%
embedding_51_87554:	?"
cin_1_87560:? "
cin_1_87562:? &
dense_layer_1_87569:	?@!
dense_layer_1_87571:@%
dense_layer_1_87573:@@!
dense_layer_1_87575:@%
dense_layer_1_87577:@!
dense_layer_1_87579:
dense_9_87584:@
dense_9_87586:
identity??-batch_normalization_1/StatefulPartitionedCall?cin_1/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?$embedding_36/StatefulPartitionedCall?$embedding_37/StatefulPartitionedCall?$embedding_38/StatefulPartitionedCall?$embedding_39/StatefulPartitionedCall?$embedding_40/StatefulPartitionedCall?$embedding_41/StatefulPartitionedCall?$embedding_42/StatefulPartitionedCall?$embedding_43/StatefulPartitionedCall?$embedding_44/StatefulPartitionedCall?$embedding_45/StatefulPartitionedCall?$embedding_46/StatefulPartitionedCall?$embedding_47/StatefulPartitionedCall?$embedding_48/StatefulPartitionedCall?$embedding_49/StatefulPartitionedCall?$embedding_50/StatefulPartitionedCall?$embedding_51/StatefulPartitionedCall? linear_1/StatefulPartitionedCall?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_1_87353batch_normalization_1_87355batch_normalization_1_87357batch_normalization_1_87359*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_852432/
-batch_normalization_1/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
 linear_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0linear_1_87370linear_1_87372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_865342"
 linear_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_87379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_853752&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_87386*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_853932&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_87393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_854112&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_87400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_854292&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_87407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_854472&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_87414*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_854652&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_87421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_854832&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_87428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_855012&
$embedding_33/StatefulPartitionedCall?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_87435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_855192&
$embedding_34/StatefulPartitionedCall?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_87442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_855372&
$embedding_35/StatefulPartitionedCall?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_87449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_855552&
$embedding_36/StatefulPartitionedCall?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_87456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_855732&
$embedding_37/StatefulPartitionedCall?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_87463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_855912&
$embedding_38/StatefulPartitionedCall?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_87470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_856092&
$embedding_39/StatefulPartitionedCall?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_87477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_856272&
$embedding_40/StatefulPartitionedCall?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_87484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_856452&
$embedding_41/StatefulPartitionedCall?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_87491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_856632&
$embedding_42/StatefulPartitionedCall?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_87498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_856812&
$embedding_43/StatefulPartitionedCall?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_87505*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_856992&
$embedding_44/StatefulPartitionedCall?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_87512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_857172&
$embedding_45/StatefulPartitionedCall?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_87519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_857352&
$embedding_46/StatefulPartitionedCall?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_87526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_857532&
$embedding_47/StatefulPartitionedCall?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_87533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_857712&
$embedding_48/StatefulPartitionedCall?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_87540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_857892&
$embedding_49/StatefulPartitionedCall?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_87547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_858072&
$embedding_50/StatefulPartitionedCall?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_87554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_858252&
$embedding_51/StatefulPartitionedCall?

packedPack-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transpose?
cin_1/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0cin_1_87560cin_1_87562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_862972
cin_1/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_layer_1_87569dense_layer_1_87571dense_layer_1_87573dense_layer_1_87575dense_layer_1_87577dense_layer_1_87579*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_861712'
%dense_layer_1/StatefulPartitionedCall?
addAddV2)linear_1/StatefulPartitionedCall:output:0&cin_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add?
add_1AddV2add:z:0.dense_layer_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0dense_9_87584dense_9_87586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_859922!
dense_9/StatefulPartitionedCally
SigmoidSigmoid(dense_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_87560*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_87560*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_87562*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_87562*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^cin_1/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall!^linear_1/StatefulPartitionedCall4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
cin_1/StatefulPartitionedCallcin_1/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall2L
$embedding_36/StatefulPartitionedCall$embedding_36/StatefulPartitionedCall2L
$embedding_37/StatefulPartitionedCall$embedding_37/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2L
$embedding_39/StatefulPartitionedCall$embedding_39/StatefulPartitionedCall2L
$embedding_40/StatefulPartitionedCall$embedding_40/StatefulPartitionedCall2L
$embedding_41/StatefulPartitionedCall$embedding_41/StatefulPartitionedCall2L
$embedding_42/StatefulPartitionedCall$embedding_42/StatefulPartitionedCall2L
$embedding_43/StatefulPartitionedCall$embedding_43/StatefulPartitionedCall2L
$embedding_44/StatefulPartitionedCall$embedding_44/StatefulPartitionedCall2L
$embedding_45/StatefulPartitionedCall$embedding_45/StatefulPartitionedCall2L
$embedding_46/StatefulPartitionedCall$embedding_46/StatefulPartitionedCall2L
$embedding_47/StatefulPartitionedCall$embedding_47/StatefulPartitionedCall2L
$embedding_48/StatefulPartitionedCall$embedding_48/StatefulPartitionedCall2L
$embedding_49/StatefulPartitionedCall$embedding_49/StatefulPartitionedCall2L
$embedding_50/StatefulPartitionedCall$embedding_50/StatefulPartitionedCall2L
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall2D
 linear_1/StatefulPartitionedCall linear_1/StatefulPartitionedCall2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
??
?(
 __inference__wrapped_model_85159
input_1L
>x_deep_fm_1_batch_normalization_1_cast_readvariableop_resource:'N
@x_deep_fm_1_batch_normalization_1_cast_1_readvariableop_resource:'N
@x_deep_fm_1_batch_normalization_1_cast_2_readvariableop_resource:'N
@x_deep_fm_1_batch_normalization_1_cast_3_readvariableop_resource:'M
;x_deep_fm_1_linear_1_dense_5_matmul_readvariableop_resource:'J
<x_deep_fm_1_linear_1_dense_5_biasadd_readvariableop_resource:A
/x_deep_fm_1_embedding_26_embedding_lookup_84801:hB
/x_deep_fm_1_embedding_27_embedding_lookup_84811:	?B
/x_deep_fm_1_embedding_28_embedding_lookup_84821:	?B
/x_deep_fm_1_embedding_29_embedding_lookup_84831:	?A
/x_deep_fm_1_embedding_30_embedding_lookup_84841:*A
/x_deep_fm_1_embedding_31_embedding_lookup_84851:	B
/x_deep_fm_1_embedding_32_embedding_lookup_84861:	?A
/x_deep_fm_1_embedding_33_embedding_lookup_84871:CA
/x_deep_fm_1_embedding_34_embedding_lookup_84881:B
/x_deep_fm_1_embedding_35_embedding_lookup_84891:	?B
/x_deep_fm_1_embedding_36_embedding_lookup_84901:	?B
/x_deep_fm_1_embedding_37_embedding_lookup_84911:	?B
/x_deep_fm_1_embedding_38_embedding_lookup_84921:	?A
/x_deep_fm_1_embedding_39_embedding_lookup_84931:B
/x_deep_fm_1_embedding_40_embedding_lookup_84941:	?
B
/x_deep_fm_1_embedding_41_embedding_lookup_84951:	?A
/x_deep_fm_1_embedding_42_embedding_lookup_84961:
B
/x_deep_fm_1_embedding_43_embedding_lookup_84971:	?B
/x_deep_fm_1_embedding_44_embedding_lookup_84981:	?A
/x_deep_fm_1_embedding_45_embedding_lookup_84991:B
/x_deep_fm_1_embedding_46_embedding_lookup_85001:	?A
/x_deep_fm_1_embedding_47_embedding_lookup_85011:
A
/x_deep_fm_1_embedding_48_embedding_lookup_85021:B
/x_deep_fm_1_embedding_49_embedding_lookup_85031:	?A
/x_deep_fm_1_embedding_50_embedding_lookup_85041:'B
/x_deep_fm_1_embedding_51_embedding_lookup_85051:	?T
=x_deep_fm_1_cin_1_conv1d_expanddims_1_readvariableop_resource:? V
?x_deep_fm_1_cin_1_conv1d_1_expanddims_1_readvariableop_resource:? S
@x_deep_fm_1_dense_layer_1_dense_6_matmul_readvariableop_resource:	?@O
Ax_deep_fm_1_dense_layer_1_dense_6_biasadd_readvariableop_resource:@R
@x_deep_fm_1_dense_layer_1_dense_7_matmul_readvariableop_resource:@@O
Ax_deep_fm_1_dense_layer_1_dense_7_biasadd_readvariableop_resource:@R
@x_deep_fm_1_dense_layer_1_dense_8_matmul_readvariableop_resource:@O
Ax_deep_fm_1_dense_layer_1_dense_8_biasadd_readvariableop_resource:D
2x_deep_fm_1_dense_9_matmul_readvariableop_resource:@A
3x_deep_fm_1_dense_9_biasadd_readvariableop_resource:
identity??5x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp?7x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp?7x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp?7x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp?4x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp?6x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?*x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp?)x_deep_fm_1/dense_9/MatMul/ReadVariableOp?8x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp?7x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp?8x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp?7x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp?8x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp?7x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp?)x_deep_fm_1/embedding_26/embedding_lookup?)x_deep_fm_1/embedding_27/embedding_lookup?)x_deep_fm_1/embedding_28/embedding_lookup?)x_deep_fm_1/embedding_29/embedding_lookup?)x_deep_fm_1/embedding_30/embedding_lookup?)x_deep_fm_1/embedding_31/embedding_lookup?)x_deep_fm_1/embedding_32/embedding_lookup?)x_deep_fm_1/embedding_33/embedding_lookup?)x_deep_fm_1/embedding_34/embedding_lookup?)x_deep_fm_1/embedding_35/embedding_lookup?)x_deep_fm_1/embedding_36/embedding_lookup?)x_deep_fm_1/embedding_37/embedding_lookup?)x_deep_fm_1/embedding_38/embedding_lookup?)x_deep_fm_1/embedding_39/embedding_lookup?)x_deep_fm_1/embedding_40/embedding_lookup?)x_deep_fm_1/embedding_41/embedding_lookup?)x_deep_fm_1/embedding_42/embedding_lookup?)x_deep_fm_1/embedding_43/embedding_lookup?)x_deep_fm_1/embedding_44/embedding_lookup?)x_deep_fm_1/embedding_45/embedding_lookup?)x_deep_fm_1/embedding_46/embedding_lookup?)x_deep_fm_1/embedding_47/embedding_lookup?)x_deep_fm_1/embedding_48/embedding_lookup?)x_deep_fm_1/embedding_49/embedding_lookup?)x_deep_fm_1/embedding_50/embedding_lookup?)x_deep_fm_1/embedding_51/embedding_lookup?3x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp?2x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp?
5x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOp>x_deep_fm_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:'*
dtype027
5x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp?
7x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp@x_deep_fm_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:'*
dtype029
7x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp?
7x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp@x_deep_fm_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:'*
dtype029
7x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp?
7x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp@x_deep_fm_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:'*
dtype029
7x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp?
1x_deep_fm_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1x_deep_fm_1/batch_normalization_1/batchnorm/add/y?
/x_deep_fm_1/batch_normalization_1/batchnorm/addAddV2?x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0:x_deep_fm_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:'21
/x_deep_fm_1/batch_normalization_1/batchnorm/add?
1x_deep_fm_1/batch_normalization_1/batchnorm/RsqrtRsqrt3x_deep_fm_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:'23
1x_deep_fm_1/batch_normalization_1/batchnorm/Rsqrt?
/x_deep_fm_1/batch_normalization_1/batchnorm/mulMul5x_deep_fm_1/batch_normalization_1/batchnorm/Rsqrt:y:0?x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:'21
/x_deep_fm_1/batch_normalization_1/batchnorm/mul?
1x_deep_fm_1/batch_normalization_1/batchnorm/mul_1Mulinput_13x_deep_fm_1/batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'23
1x_deep_fm_1/batch_normalization_1/batchnorm/mul_1?
1x_deep_fm_1/batch_normalization_1/batchnorm/mul_2Mul=x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp:value:03x_deep_fm_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:'23
1x_deep_fm_1/batch_normalization_1/batchnorm/mul_2?
/x_deep_fm_1/batch_normalization_1/batchnorm/subSub?x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp:value:05x_deep_fm_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:'21
/x_deep_fm_1/batch_normalization_1/batchnorm/sub?
1x_deep_fm_1/batch_normalization_1/batchnorm/add_1AddV25x_deep_fm_1/batch_normalization_1/batchnorm/mul_1:z:03x_deep_fm_1/batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'23
1x_deep_fm_1/batch_normalization_1/batchnorm/add_1?
x_deep_fm_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
x_deep_fm_1/strided_slice/stack?
!x_deep_fm_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice/stack_1?
!x_deep_fm_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!x_deep_fm_1/strided_slice/stack_2?
x_deep_fm_1/strided_sliceStridedSlice5x_deep_fm_1/batch_normalization_1/batchnorm/add_1:z:0(x_deep_fm_1/strided_slice/stack:output:0*x_deep_fm_1/strided_slice/stack_1:output:0*x_deep_fm_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
x_deep_fm_1/strided_slice?
!x_deep_fm_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_1/stack?
#x_deep_fm_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2%
#x_deep_fm_1/strided_slice_1/stack_1?
#x_deep_fm_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_1/stack_2?
x_deep_fm_1/strided_slice_1StridedSlice5x_deep_fm_1/batch_normalization_1/batchnorm/add_1:z:0*x_deep_fm_1/strided_slice_1/stack:output:0,x_deep_fm_1/strided_slice_1/stack_1:output:0,x_deep_fm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
x_deep_fm_1/strided_slice_1?
2x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOpReadVariableOp;x_deep_fm_1_linear_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype024
2x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp?
#x_deep_fm_1/linear_1/dense_5/MatMulMatMul5x_deep_fm_1/batch_normalization_1/batchnorm/add_1:z:0:x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2%
#x_deep_fm_1/linear_1/dense_5/MatMul?
3x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp<x_deep_fm_1_linear_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp?
$x_deep_fm_1/linear_1/dense_5/BiasAddBiasAdd-x_deep_fm_1/linear_1/dense_5/MatMul:product:0;x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$x_deep_fm_1/linear_1/dense_5/BiasAdd?
!x_deep_fm_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!x_deep_fm_1/strided_slice_2/stack?
#x_deep_fm_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_2/stack_1?
#x_deep_fm_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_2/stack_2?
x_deep_fm_1/strided_slice_2StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_2/stack:output:0,x_deep_fm_1/strided_slice_2/stack_1:output:0,x_deep_fm_1/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_2?
x_deep_fm_1/embedding_26/CastCast$x_deep_fm_1/strided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_26/Cast?
)x_deep_fm_1/embedding_26/embedding_lookupResourceGather/x_deep_fm_1_embedding_26_embedding_lookup_84801!x_deep_fm_1/embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_26/embedding_lookup/84801*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_26/embedding_lookup?
2x_deep_fm_1/embedding_26/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_26/embedding_lookup/84801*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_26/embedding_lookup/Identity?
4x_deep_fm_1/embedding_26/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_26/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_3/stack?
#x_deep_fm_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_3/stack_1?
#x_deep_fm_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_3/stack_2?
x_deep_fm_1/strided_slice_3StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_3/stack:output:0,x_deep_fm_1/strided_slice_3/stack_1:output:0,x_deep_fm_1/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_3?
x_deep_fm_1/embedding_27/CastCast$x_deep_fm_1/strided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_27/Cast?
)x_deep_fm_1/embedding_27/embedding_lookupResourceGather/x_deep_fm_1_embedding_27_embedding_lookup_84811!x_deep_fm_1/embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_27/embedding_lookup/84811*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_27/embedding_lookup?
2x_deep_fm_1/embedding_27/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_27/embedding_lookup/84811*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_27/embedding_lookup/Identity?
4x_deep_fm_1/embedding_27/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_27/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_4/stack?
#x_deep_fm_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_4/stack_1?
#x_deep_fm_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_4/stack_2?
x_deep_fm_1/strided_slice_4StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_4/stack:output:0,x_deep_fm_1/strided_slice_4/stack_1:output:0,x_deep_fm_1/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_4?
x_deep_fm_1/embedding_28/CastCast$x_deep_fm_1/strided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_28/Cast?
)x_deep_fm_1/embedding_28/embedding_lookupResourceGather/x_deep_fm_1_embedding_28_embedding_lookup_84821!x_deep_fm_1/embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_28/embedding_lookup/84821*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_28/embedding_lookup?
2x_deep_fm_1/embedding_28/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_28/embedding_lookup/84821*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_28/embedding_lookup/Identity?
4x_deep_fm_1/embedding_28/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_28/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_5/stack?
#x_deep_fm_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_5/stack_1?
#x_deep_fm_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_5/stack_2?
x_deep_fm_1/strided_slice_5StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_5/stack:output:0,x_deep_fm_1/strided_slice_5/stack_1:output:0,x_deep_fm_1/strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_5?
x_deep_fm_1/embedding_29/CastCast$x_deep_fm_1/strided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_29/Cast?
)x_deep_fm_1/embedding_29/embedding_lookupResourceGather/x_deep_fm_1_embedding_29_embedding_lookup_84831!x_deep_fm_1/embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_29/embedding_lookup/84831*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_29/embedding_lookup?
2x_deep_fm_1/embedding_29/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_29/embedding_lookup/84831*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_29/embedding_lookup/Identity?
4x_deep_fm_1/embedding_29/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_29/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_6/stack?
#x_deep_fm_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_6/stack_1?
#x_deep_fm_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_6/stack_2?
x_deep_fm_1/strided_slice_6StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_6/stack:output:0,x_deep_fm_1/strided_slice_6/stack_1:output:0,x_deep_fm_1/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_6?
x_deep_fm_1/embedding_30/CastCast$x_deep_fm_1/strided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_30/Cast?
)x_deep_fm_1/embedding_30/embedding_lookupResourceGather/x_deep_fm_1_embedding_30_embedding_lookup_84841!x_deep_fm_1/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_30/embedding_lookup/84841*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_30/embedding_lookup?
2x_deep_fm_1/embedding_30/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_30/embedding_lookup/84841*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_30/embedding_lookup/Identity?
4x_deep_fm_1/embedding_30/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_30/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_7/stack?
#x_deep_fm_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_7/stack_1?
#x_deep_fm_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_7/stack_2?
x_deep_fm_1/strided_slice_7StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_7/stack:output:0,x_deep_fm_1/strided_slice_7/stack_1:output:0,x_deep_fm_1/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_7?
x_deep_fm_1/embedding_31/CastCast$x_deep_fm_1/strided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_31/Cast?
)x_deep_fm_1/embedding_31/embedding_lookupResourceGather/x_deep_fm_1_embedding_31_embedding_lookup_84851!x_deep_fm_1/embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_31/embedding_lookup/84851*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_31/embedding_lookup?
2x_deep_fm_1/embedding_31/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_31/embedding_lookup/84851*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_31/embedding_lookup/Identity?
4x_deep_fm_1/embedding_31/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_31/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_8/stack?
#x_deep_fm_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_8/stack_1?
#x_deep_fm_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_8/stack_2?
x_deep_fm_1/strided_slice_8StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_8/stack:output:0,x_deep_fm_1/strided_slice_8/stack_1:output:0,x_deep_fm_1/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_8?
x_deep_fm_1/embedding_32/CastCast$x_deep_fm_1/strided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_32/Cast?
)x_deep_fm_1/embedding_32/embedding_lookupResourceGather/x_deep_fm_1_embedding_32_embedding_lookup_84861!x_deep_fm_1/embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_32/embedding_lookup/84861*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_32/embedding_lookup?
2x_deep_fm_1/embedding_32/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_32/embedding_lookup/84861*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_32/embedding_lookup/Identity?
4x_deep_fm_1/embedding_32/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_32/embedding_lookup/Identity_1?
!x_deep_fm_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2#
!x_deep_fm_1/strided_slice_9/stack?
#x_deep_fm_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#x_deep_fm_1/strided_slice_9/stack_1?
#x_deep_fm_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#x_deep_fm_1/strided_slice_9/stack_2?
x_deep_fm_1/strided_slice_9StridedSlice$x_deep_fm_1/strided_slice_1:output:0*x_deep_fm_1/strided_slice_9/stack:output:0,x_deep_fm_1/strided_slice_9/stack_1:output:0,x_deep_fm_1/strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_9?
x_deep_fm_1/embedding_33/CastCast$x_deep_fm_1/strided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_33/Cast?
)x_deep_fm_1/embedding_33/embedding_lookupResourceGather/x_deep_fm_1_embedding_33_embedding_lookup_84871!x_deep_fm_1/embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_33/embedding_lookup/84871*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_33/embedding_lookup?
2x_deep_fm_1/embedding_33/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_33/embedding_lookup/84871*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_33/embedding_lookup/Identity?
4x_deep_fm_1/embedding_33/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_33/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_10/stack?
$x_deep_fm_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2&
$x_deep_fm_1/strided_slice_10/stack_1?
$x_deep_fm_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_10/stack_2?
x_deep_fm_1/strided_slice_10StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_10/stack:output:0-x_deep_fm_1/strided_slice_10/stack_1:output:0-x_deep_fm_1/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_10?
x_deep_fm_1/embedding_34/CastCast%x_deep_fm_1/strided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_34/Cast?
)x_deep_fm_1/embedding_34/embedding_lookupResourceGather/x_deep_fm_1_embedding_34_embedding_lookup_84881!x_deep_fm_1/embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_34/embedding_lookup/84881*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_34/embedding_lookup?
2x_deep_fm_1/embedding_34/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_34/embedding_lookup/84881*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_34/embedding_lookup/Identity?
4x_deep_fm_1/embedding_34/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_34/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2$
"x_deep_fm_1/strided_slice_11/stack?
$x_deep_fm_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2&
$x_deep_fm_1/strided_slice_11/stack_1?
$x_deep_fm_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_11/stack_2?
x_deep_fm_1/strided_slice_11StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_11/stack:output:0-x_deep_fm_1/strided_slice_11/stack_1:output:0-x_deep_fm_1/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_11?
x_deep_fm_1/embedding_35/CastCast%x_deep_fm_1/strided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_35/Cast?
)x_deep_fm_1/embedding_35/embedding_lookupResourceGather/x_deep_fm_1_embedding_35_embedding_lookup_84891!x_deep_fm_1/embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_35/embedding_lookup/84891*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_35/embedding_lookup?
2x_deep_fm_1/embedding_35/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_35/embedding_lookup/84891*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_35/embedding_lookup/Identity?
4x_deep_fm_1/embedding_35/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_35/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2$
"x_deep_fm_1/strided_slice_12/stack?
$x_deep_fm_1/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_12/stack_1?
$x_deep_fm_1/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_12/stack_2?
x_deep_fm_1/strided_slice_12StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_12/stack:output:0-x_deep_fm_1/strided_slice_12/stack_1:output:0-x_deep_fm_1/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_12?
x_deep_fm_1/embedding_36/CastCast%x_deep_fm_1/strided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_36/Cast?
)x_deep_fm_1/embedding_36/embedding_lookupResourceGather/x_deep_fm_1_embedding_36_embedding_lookup_84901!x_deep_fm_1/embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_36/embedding_lookup/84901*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_36/embedding_lookup?
2x_deep_fm_1/embedding_36/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_36/embedding_lookup/84901*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_36/embedding_lookup/Identity?
4x_deep_fm_1/embedding_36/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_36/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_13/stack?
$x_deep_fm_1/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_13/stack_1?
$x_deep_fm_1/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_13/stack_2?
x_deep_fm_1/strided_slice_13StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_13/stack:output:0-x_deep_fm_1/strided_slice_13/stack_1:output:0-x_deep_fm_1/strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_13?
x_deep_fm_1/embedding_37/CastCast%x_deep_fm_1/strided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_37/Cast?
)x_deep_fm_1/embedding_37/embedding_lookupResourceGather/x_deep_fm_1_embedding_37_embedding_lookup_84911!x_deep_fm_1/embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_37/embedding_lookup/84911*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_37/embedding_lookup?
2x_deep_fm_1/embedding_37/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_37/embedding_lookup/84911*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_37/embedding_lookup/Identity?
4x_deep_fm_1/embedding_37/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_37/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_14/stack?
$x_deep_fm_1/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_14/stack_1?
$x_deep_fm_1/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_14/stack_2?
x_deep_fm_1/strided_slice_14StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_14/stack:output:0-x_deep_fm_1/strided_slice_14/stack_1:output:0-x_deep_fm_1/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_14?
x_deep_fm_1/embedding_38/CastCast%x_deep_fm_1/strided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_38/Cast?
)x_deep_fm_1/embedding_38/embedding_lookupResourceGather/x_deep_fm_1_embedding_38_embedding_lookup_84921!x_deep_fm_1/embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_38/embedding_lookup/84921*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_38/embedding_lookup?
2x_deep_fm_1/embedding_38/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_38/embedding_lookup/84921*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_38/embedding_lookup/Identity?
4x_deep_fm_1/embedding_38/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_38/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_15/stack?
$x_deep_fm_1/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_15/stack_1?
$x_deep_fm_1/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_15/stack_2?
x_deep_fm_1/strided_slice_15StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_15/stack:output:0-x_deep_fm_1/strided_slice_15/stack_1:output:0-x_deep_fm_1/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_15?
x_deep_fm_1/embedding_39/CastCast%x_deep_fm_1/strided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_39/Cast?
)x_deep_fm_1/embedding_39/embedding_lookupResourceGather/x_deep_fm_1_embedding_39_embedding_lookup_84931!x_deep_fm_1/embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_39/embedding_lookup/84931*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_39/embedding_lookup?
2x_deep_fm_1/embedding_39/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_39/embedding_lookup/84931*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_39/embedding_lookup/Identity?
4x_deep_fm_1/embedding_39/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_39/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_16/stack?
$x_deep_fm_1/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_16/stack_1?
$x_deep_fm_1/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_16/stack_2?
x_deep_fm_1/strided_slice_16StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_16/stack:output:0-x_deep_fm_1/strided_slice_16/stack_1:output:0-x_deep_fm_1/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_16?
x_deep_fm_1/embedding_40/CastCast%x_deep_fm_1/strided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_40/Cast?
)x_deep_fm_1/embedding_40/embedding_lookupResourceGather/x_deep_fm_1_embedding_40_embedding_lookup_84941!x_deep_fm_1/embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_40/embedding_lookup/84941*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_40/embedding_lookup?
2x_deep_fm_1/embedding_40/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_40/embedding_lookup/84941*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_40/embedding_lookup/Identity?
4x_deep_fm_1/embedding_40/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_40/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_17/stack?
$x_deep_fm_1/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_17/stack_1?
$x_deep_fm_1/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_17/stack_2?
x_deep_fm_1/strided_slice_17StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_17/stack:output:0-x_deep_fm_1/strided_slice_17/stack_1:output:0-x_deep_fm_1/strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_17?
x_deep_fm_1/embedding_41/CastCast%x_deep_fm_1/strided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_41/Cast?
)x_deep_fm_1/embedding_41/embedding_lookupResourceGather/x_deep_fm_1_embedding_41_embedding_lookup_84951!x_deep_fm_1/embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_41/embedding_lookup/84951*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_41/embedding_lookup?
2x_deep_fm_1/embedding_41/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_41/embedding_lookup/84951*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_41/embedding_lookup/Identity?
4x_deep_fm_1/embedding_41/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_41/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_18/stack?
$x_deep_fm_1/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_18/stack_1?
$x_deep_fm_1/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_18/stack_2?
x_deep_fm_1/strided_slice_18StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_18/stack:output:0-x_deep_fm_1/strided_slice_18/stack_1:output:0-x_deep_fm_1/strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_18?
x_deep_fm_1/embedding_42/CastCast%x_deep_fm_1/strided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_42/Cast?
)x_deep_fm_1/embedding_42/embedding_lookupResourceGather/x_deep_fm_1_embedding_42_embedding_lookup_84961!x_deep_fm_1/embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_42/embedding_lookup/84961*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_42/embedding_lookup?
2x_deep_fm_1/embedding_42/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_42/embedding_lookup/84961*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_42/embedding_lookup/Identity?
4x_deep_fm_1/embedding_42/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_42/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_19/stack?
$x_deep_fm_1/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_19/stack_1?
$x_deep_fm_1/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_19/stack_2?
x_deep_fm_1/strided_slice_19StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_19/stack:output:0-x_deep_fm_1/strided_slice_19/stack_1:output:0-x_deep_fm_1/strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_19?
x_deep_fm_1/embedding_43/CastCast%x_deep_fm_1/strided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_43/Cast?
)x_deep_fm_1/embedding_43/embedding_lookupResourceGather/x_deep_fm_1_embedding_43_embedding_lookup_84971!x_deep_fm_1/embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_43/embedding_lookup/84971*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_43/embedding_lookup?
2x_deep_fm_1/embedding_43/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_43/embedding_lookup/84971*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_43/embedding_lookup/Identity?
4x_deep_fm_1/embedding_43/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_43/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_20/stack?
$x_deep_fm_1/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_20/stack_1?
$x_deep_fm_1/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_20/stack_2?
x_deep_fm_1/strided_slice_20StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_20/stack:output:0-x_deep_fm_1/strided_slice_20/stack_1:output:0-x_deep_fm_1/strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_20?
x_deep_fm_1/embedding_44/CastCast%x_deep_fm_1/strided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_44/Cast?
)x_deep_fm_1/embedding_44/embedding_lookupResourceGather/x_deep_fm_1_embedding_44_embedding_lookup_84981!x_deep_fm_1/embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_44/embedding_lookup/84981*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_44/embedding_lookup?
2x_deep_fm_1/embedding_44/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_44/embedding_lookup/84981*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_44/embedding_lookup/Identity?
4x_deep_fm_1/embedding_44/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_44/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_21/stack?
$x_deep_fm_1/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_21/stack_1?
$x_deep_fm_1/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_21/stack_2?
x_deep_fm_1/strided_slice_21StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_21/stack:output:0-x_deep_fm_1/strided_slice_21/stack_1:output:0-x_deep_fm_1/strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_21?
x_deep_fm_1/embedding_45/CastCast%x_deep_fm_1/strided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_45/Cast?
)x_deep_fm_1/embedding_45/embedding_lookupResourceGather/x_deep_fm_1_embedding_45_embedding_lookup_84991!x_deep_fm_1/embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_45/embedding_lookup/84991*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_45/embedding_lookup?
2x_deep_fm_1/embedding_45/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_45/embedding_lookup/84991*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_45/embedding_lookup/Identity?
4x_deep_fm_1/embedding_45/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_45/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_22/stack?
$x_deep_fm_1/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_22/stack_1?
$x_deep_fm_1/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_22/stack_2?
x_deep_fm_1/strided_slice_22StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_22/stack:output:0-x_deep_fm_1/strided_slice_22/stack_1:output:0-x_deep_fm_1/strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_22?
x_deep_fm_1/embedding_46/CastCast%x_deep_fm_1/strided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_46/Cast?
)x_deep_fm_1/embedding_46/embedding_lookupResourceGather/x_deep_fm_1_embedding_46_embedding_lookup_85001!x_deep_fm_1/embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_46/embedding_lookup/85001*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_46/embedding_lookup?
2x_deep_fm_1/embedding_46/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_46/embedding_lookup/85001*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_46/embedding_lookup/Identity?
4x_deep_fm_1/embedding_46/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_46/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_23/stack?
$x_deep_fm_1/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_23/stack_1?
$x_deep_fm_1/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_23/stack_2?
x_deep_fm_1/strided_slice_23StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_23/stack:output:0-x_deep_fm_1/strided_slice_23/stack_1:output:0-x_deep_fm_1/strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_23?
x_deep_fm_1/embedding_47/CastCast%x_deep_fm_1/strided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_47/Cast?
)x_deep_fm_1/embedding_47/embedding_lookupResourceGather/x_deep_fm_1_embedding_47_embedding_lookup_85011!x_deep_fm_1/embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_47/embedding_lookup/85011*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_47/embedding_lookup?
2x_deep_fm_1/embedding_47/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_47/embedding_lookup/85011*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_47/embedding_lookup/Identity?
4x_deep_fm_1/embedding_47/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_47/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_24/stack?
$x_deep_fm_1/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_24/stack_1?
$x_deep_fm_1/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_24/stack_2?
x_deep_fm_1/strided_slice_24StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_24/stack:output:0-x_deep_fm_1/strided_slice_24/stack_1:output:0-x_deep_fm_1/strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_24?
x_deep_fm_1/embedding_48/CastCast%x_deep_fm_1/strided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_48/Cast?
)x_deep_fm_1/embedding_48/embedding_lookupResourceGather/x_deep_fm_1_embedding_48_embedding_lookup_85021!x_deep_fm_1/embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_48/embedding_lookup/85021*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_48/embedding_lookup?
2x_deep_fm_1/embedding_48/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_48/embedding_lookup/85021*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_48/embedding_lookup/Identity?
4x_deep_fm_1/embedding_48/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_48/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_25/stack?
$x_deep_fm_1/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_25/stack_1?
$x_deep_fm_1/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_25/stack_2?
x_deep_fm_1/strided_slice_25StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_25/stack:output:0-x_deep_fm_1/strided_slice_25/stack_1:output:0-x_deep_fm_1/strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_25?
x_deep_fm_1/embedding_49/CastCast%x_deep_fm_1/strided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_49/Cast?
)x_deep_fm_1/embedding_49/embedding_lookupResourceGather/x_deep_fm_1_embedding_49_embedding_lookup_85031!x_deep_fm_1/embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_49/embedding_lookup/85031*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_49/embedding_lookup?
2x_deep_fm_1/embedding_49/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_49/embedding_lookup/85031*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_49/embedding_lookup/Identity?
4x_deep_fm_1/embedding_49/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_49/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_26/stack?
$x_deep_fm_1/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_26/stack_1?
$x_deep_fm_1/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_26/stack_2?
x_deep_fm_1/strided_slice_26StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_26/stack:output:0-x_deep_fm_1/strided_slice_26/stack_1:output:0-x_deep_fm_1/strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_26?
x_deep_fm_1/embedding_50/CastCast%x_deep_fm_1/strided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_50/Cast?
)x_deep_fm_1/embedding_50/embedding_lookupResourceGather/x_deep_fm_1_embedding_50_embedding_lookup_85041!x_deep_fm_1/embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_50/embedding_lookup/85041*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_50/embedding_lookup?
2x_deep_fm_1/embedding_50/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_50/embedding_lookup/85041*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_50/embedding_lookup/Identity?
4x_deep_fm_1/embedding_50/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_50/embedding_lookup/Identity_1?
"x_deep_fm_1/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2$
"x_deep_fm_1/strided_slice_27/stack?
$x_deep_fm_1/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$x_deep_fm_1/strided_slice_27/stack_1?
$x_deep_fm_1/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$x_deep_fm_1/strided_slice_27/stack_2?
x_deep_fm_1/strided_slice_27StridedSlice$x_deep_fm_1/strided_slice_1:output:0+x_deep_fm_1/strided_slice_27/stack:output:0-x_deep_fm_1/strided_slice_27/stack_1:output:0-x_deep_fm_1/strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
x_deep_fm_1/strided_slice_27?
x_deep_fm_1/embedding_51/CastCast%x_deep_fm_1/strided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
x_deep_fm_1/embedding_51/Cast?
)x_deep_fm_1/embedding_51/embedding_lookupResourceGather/x_deep_fm_1_embedding_51_embedding_lookup_85051!x_deep_fm_1/embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*B
_class8
64loc:@x_deep_fm_1/embedding_51/embedding_lookup/85051*'
_output_shapes
:?????????*
dtype02+
)x_deep_fm_1/embedding_51/embedding_lookup?
2x_deep_fm_1/embedding_51/embedding_lookup/IdentityIdentity2x_deep_fm_1/embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*B
_class8
64loc:@x_deep_fm_1/embedding_51/embedding_lookup/85051*'
_output_shapes
:?????????24
2x_deep_fm_1/embedding_51/embedding_lookup/Identity?
4x_deep_fm_1/embedding_51/embedding_lookup/Identity_1Identity;x_deep_fm_1/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????26
4x_deep_fm_1/embedding_51/embedding_lookup/Identity_1?
x_deep_fm_1/packedPack=x_deep_fm_1/embedding_26/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_27/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_28/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_29/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_30/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_31/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_32/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_33/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_34/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_35/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_36/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_37/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_38/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_39/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_40/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_41/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_42/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_43/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_44/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_45/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_46/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_47/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_48/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_49/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_50/embedding_lookup/Identity_1:output:0=x_deep_fm_1/embedding_51/embedding_lookup/Identity_1:output:0*
N*
T0*+
_output_shapes
:?????????2
x_deep_fm_1/packed?
x_deep_fm_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
x_deep_fm_1/transpose/perm?
x_deep_fm_1/transpose	Transposex_deep_fm_1/packed:output:0#x_deep_fm_1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
x_deep_fm_1/transpose?
!x_deep_fm_1/cin_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!x_deep_fm_1/cin_1/split/split_dim?
x_deep_fm_1/cin_1/splitSplit*x_deep_fm_1/cin_1/split/split_dim:output:0x_deep_fm_1/transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
x_deep_fm_1/cin_1/split?
#x_deep_fm_1/cin_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#x_deep_fm_1/cin_1/split_1/split_dim?
x_deep_fm_1/cin_1/split_1Split,x_deep_fm_1/cin_1/split_1/split_dim:output:0x_deep_fm_1/transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
x_deep_fm_1/cin_1/split_1?
x_deep_fm_1/cin_1/MatMul/aPack x_deep_fm_1/cin_1/split:output:0 x_deep_fm_1/cin_1/split:output:1 x_deep_fm_1/cin_1/split:output:2 x_deep_fm_1/cin_1/split:output:3 x_deep_fm_1/cin_1/split:output:4 x_deep_fm_1/cin_1/split:output:5 x_deep_fm_1/cin_1/split:output:6 x_deep_fm_1/cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
x_deep_fm_1/cin_1/MatMul/a?
x_deep_fm_1/cin_1/MatMul/bPack"x_deep_fm_1/cin_1/split_1:output:0"x_deep_fm_1/cin_1/split_1:output:1"x_deep_fm_1/cin_1/split_1:output:2"x_deep_fm_1/cin_1/split_1:output:3"x_deep_fm_1/cin_1/split_1:output:4"x_deep_fm_1/cin_1/split_1:output:5"x_deep_fm_1/cin_1/split_1:output:6"x_deep_fm_1/cin_1/split_1:output:7*
N*
T0*/
_output_shapes
:?????????2
x_deep_fm_1/cin_1/MatMul/b?
x_deep_fm_1/cin_1/MatMulBatchMatMulV2#x_deep_fm_1/cin_1/MatMul/a:output:0#x_deep_fm_1/cin_1/MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
x_deep_fm_1/cin_1/MatMul?
x_deep_fm_1/cin_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2!
x_deep_fm_1/cin_1/Reshape/shape?
x_deep_fm_1/cin_1/ReshapeReshape!x_deep_fm_1/cin_1/MatMul:output:0(x_deep_fm_1/cin_1/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
x_deep_fm_1/cin_1/Reshape?
 x_deep_fm_1/cin_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 x_deep_fm_1/cin_1/transpose/perm?
x_deep_fm_1/cin_1/transpose	Transpose"x_deep_fm_1/cin_1/Reshape:output:0)x_deep_fm_1/cin_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
x_deep_fm_1/cin_1/transpose?
'x_deep_fm_1/cin_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'x_deep_fm_1/cin_1/conv1d/ExpandDims/dim?
#x_deep_fm_1/cin_1/conv1d/ExpandDims
ExpandDimsx_deep_fm_1/cin_1/transpose:y:00x_deep_fm_1/cin_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2%
#x_deep_fm_1/cin_1/conv1d/ExpandDims?
4x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=x_deep_fm_1_cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype026
4x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp?
)x_deep_fm_1/cin_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)x_deep_fm_1/cin_1/conv1d/ExpandDims_1/dim?
%x_deep_fm_1/cin_1/conv1d/ExpandDims_1
ExpandDims<x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp:value:02x_deep_fm_1/cin_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2'
%x_deep_fm_1/cin_1/conv1d/ExpandDims_1?
x_deep_fm_1/cin_1/conv1dConv2D,x_deep_fm_1/cin_1/conv1d/ExpandDims:output:0.x_deep_fm_1/cin_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
x_deep_fm_1/cin_1/conv1d?
 x_deep_fm_1/cin_1/conv1d/SqueezeSqueeze!x_deep_fm_1/cin_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2"
 x_deep_fm_1/cin_1/conv1d/Squeeze?
"x_deep_fm_1/cin_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"x_deep_fm_1/cin_1/transpose_1/perm?
x_deep_fm_1/cin_1/transpose_1	Transpose)x_deep_fm_1/cin_1/conv1d/Squeeze:output:0+x_deep_fm_1/cin_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
x_deep_fm_1/cin_1/transpose_1?
#x_deep_fm_1/cin_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#x_deep_fm_1/cin_1/split_2/split_dim?
x_deep_fm_1/cin_1/split_2Split,x_deep_fm_1/cin_1/split_2/split_dim:output:0!x_deep_fm_1/cin_1/transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2
x_deep_fm_1/cin_1/split_2?
x_deep_fm_1/cin_1/MatMul_1/aPack x_deep_fm_1/cin_1/split:output:0 x_deep_fm_1/cin_1/split:output:1 x_deep_fm_1/cin_1/split:output:2 x_deep_fm_1/cin_1/split:output:3 x_deep_fm_1/cin_1/split:output:4 x_deep_fm_1/cin_1/split:output:5 x_deep_fm_1/cin_1/split:output:6 x_deep_fm_1/cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
x_deep_fm_1/cin_1/MatMul_1/a?
x_deep_fm_1/cin_1/MatMul_1/bPack"x_deep_fm_1/cin_1/split_2:output:0"x_deep_fm_1/cin_1/split_2:output:1"x_deep_fm_1/cin_1/split_2:output:2"x_deep_fm_1/cin_1/split_2:output:3"x_deep_fm_1/cin_1/split_2:output:4"x_deep_fm_1/cin_1/split_2:output:5"x_deep_fm_1/cin_1/split_2:output:6"x_deep_fm_1/cin_1/split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2
x_deep_fm_1/cin_1/MatMul_1/b?
x_deep_fm_1/cin_1/MatMul_1BatchMatMulV2%x_deep_fm_1/cin_1/MatMul_1/a:output:0%x_deep_fm_1/cin_1/MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2
x_deep_fm_1/cin_1/MatMul_1?
!x_deep_fm_1/cin_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2#
!x_deep_fm_1/cin_1/Reshape_1/shape?
x_deep_fm_1/cin_1/Reshape_1Reshape#x_deep_fm_1/cin_1/MatMul_1:output:0*x_deep_fm_1/cin_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
x_deep_fm_1/cin_1/Reshape_1?
"x_deep_fm_1/cin_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"x_deep_fm_1/cin_1/transpose_2/perm?
x_deep_fm_1/cin_1/transpose_2	Transpose$x_deep_fm_1/cin_1/Reshape_1:output:0+x_deep_fm_1/cin_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
x_deep_fm_1/cin_1/transpose_2?
)x_deep_fm_1/cin_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)x_deep_fm_1/cin_1/conv1d_1/ExpandDims/dim?
%x_deep_fm_1/cin_1/conv1d_1/ExpandDims
ExpandDims!x_deep_fm_1/cin_1/transpose_2:y:02x_deep_fm_1/cin_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2'
%x_deep_fm_1/cin_1/conv1d_1/ExpandDims?
6x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp?x_deep_fm_1_cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?
+x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/dim?
'x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1
ExpandDims>x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:04x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1?
x_deep_fm_1/cin_1/conv1d_1Conv2D.x_deep_fm_1/cin_1/conv1d_1/ExpandDims:output:00x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
x_deep_fm_1/cin_1/conv1d_1?
"x_deep_fm_1/cin_1/conv1d_1/SqueezeSqueeze#x_deep_fm_1/cin_1/conv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2$
"x_deep_fm_1/cin_1/conv1d_1/Squeeze?
"x_deep_fm_1/cin_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"x_deep_fm_1/cin_1/transpose_3/perm?
x_deep_fm_1/cin_1/transpose_3	Transpose+x_deep_fm_1/cin_1/conv1d_1/Squeeze:output:0+x_deep_fm_1/cin_1/transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
x_deep_fm_1/cin_1/transpose_3?
x_deep_fm_1/cin_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
x_deep_fm_1/cin_1/concat/axis?
x_deep_fm_1/cin_1/concatConcatV2!x_deep_fm_1/cin_1/transpose_1:y:0!x_deep_fm_1/cin_1/transpose_3:y:0&x_deep_fm_1/cin_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
x_deep_fm_1/cin_1/concat?
'x_deep_fm_1/cin_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'x_deep_fm_1/cin_1/Sum/reduction_indices?
x_deep_fm_1/cin_1/SumSum!x_deep_fm_1/cin_1/concat:output:00x_deep_fm_1/cin_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
x_deep_fm_1/cin_1/Sum?
x_deep_fm_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
x_deep_fm_1/Reshape/shape?
x_deep_fm_1/ReshapeReshapex_deep_fm_1/transpose:y:0"x_deep_fm_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2
x_deep_fm_1/Reshapet
x_deep_fm_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
x_deep_fm_1/concat/axis?
x_deep_fm_1/concatConcatV2"x_deep_fm_1/strided_slice:output:0x_deep_fm_1/Reshape:output:0 x_deep_fm_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
x_deep_fm_1/concat?
7x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOp@x_deep_fm_1_dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype029
7x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp?
(x_deep_fm_1/dense_layer_1/dense_6/MatMulMatMulx_deep_fm_1/concat:output:0?x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(x_deep_fm_1/dense_layer_1/dense_6/MatMul?
8x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOpAx_deep_fm_1_dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp?
)x_deep_fm_1/dense_layer_1/dense_6/BiasAddBiasAdd2x_deep_fm_1/dense_layer_1/dense_6/MatMul:product:0@x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2+
)x_deep_fm_1/dense_layer_1/dense_6/BiasAdd?
&x_deep_fm_1/dense_layer_1/dense_6/ReluRelu2x_deep_fm_1/dense_layer_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2(
&x_deep_fm_1/dense_layer_1/dense_6/Relu?
7x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOpReadVariableOp@x_deep_fm_1_dense_layer_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype029
7x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp?
(x_deep_fm_1/dense_layer_1/dense_7/MatMulMatMul4x_deep_fm_1/dense_layer_1/dense_6/Relu:activations:0?x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2*
(x_deep_fm_1/dense_layer_1/dense_7/MatMul?
8x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpAx_deep_fm_1_dense_layer_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp?
)x_deep_fm_1/dense_layer_1/dense_7/BiasAddBiasAdd2x_deep_fm_1/dense_layer_1/dense_7/MatMul:product:0@x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2+
)x_deep_fm_1/dense_layer_1/dense_7/BiasAdd?
&x_deep_fm_1/dense_layer_1/dense_7/ReluRelu2x_deep_fm_1/dense_layer_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2(
&x_deep_fm_1/dense_layer_1/dense_7/Relu?
,x_deep_fm_1/dense_layer_1/dropout_1/IdentityIdentity4x_deep_fm_1/dense_layer_1/dense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2.
,x_deep_fm_1/dense_layer_1/dropout_1/Identity?
7x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOpReadVariableOp@x_deep_fm_1_dense_layer_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype029
7x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp?
(x_deep_fm_1/dense_layer_1/dense_8/MatMulMatMul5x_deep_fm_1/dense_layer_1/dropout_1/Identity:output:0?x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(x_deep_fm_1/dense_layer_1/dense_8/MatMul?
8x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOpReadVariableOpAx_deep_fm_1_dense_layer_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp?
)x_deep_fm_1/dense_layer_1/dense_8/BiasAddBiasAdd2x_deep_fm_1/dense_layer_1/dense_8/MatMul:product:0@x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)x_deep_fm_1/dense_layer_1/dense_8/BiasAdd?
x_deep_fm_1/addAddV2-x_deep_fm_1/linear_1/dense_5/BiasAdd:output:0x_deep_fm_1/cin_1/Sum:output:0*
T0*'
_output_shapes
:?????????@2
x_deep_fm_1/add?
x_deep_fm_1/add_1AddV2x_deep_fm_1/add:z:02x_deep_fm_1/dense_layer_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
x_deep_fm_1/add_1?
)x_deep_fm_1/dense_9/MatMul/ReadVariableOpReadVariableOp2x_deep_fm_1_dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02+
)x_deep_fm_1/dense_9/MatMul/ReadVariableOp?
x_deep_fm_1/dense_9/MatMulMatMulx_deep_fm_1/add_1:z:01x_deep_fm_1/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
x_deep_fm_1/dense_9/MatMul?
*x_deep_fm_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp3x_deep_fm_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp?
x_deep_fm_1/dense_9/BiasAddBiasAdd$x_deep_fm_1/dense_9/MatMul:product:02x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
x_deep_fm_1/dense_9/BiasAdd?
x_deep_fm_1/SigmoidSigmoid$x_deep_fm_1/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
x_deep_fm_1/Sigmoidr
IdentityIdentityx_deep_fm_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp6^x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp8^x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp8^x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp8^x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp5^x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp7^x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp+^x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp*^x_deep_fm_1/dense_9/MatMul/ReadVariableOp9^x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp8^x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp9^x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp8^x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp9^x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp8^x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp*^x_deep_fm_1/embedding_26/embedding_lookup*^x_deep_fm_1/embedding_27/embedding_lookup*^x_deep_fm_1/embedding_28/embedding_lookup*^x_deep_fm_1/embedding_29/embedding_lookup*^x_deep_fm_1/embedding_30/embedding_lookup*^x_deep_fm_1/embedding_31/embedding_lookup*^x_deep_fm_1/embedding_32/embedding_lookup*^x_deep_fm_1/embedding_33/embedding_lookup*^x_deep_fm_1/embedding_34/embedding_lookup*^x_deep_fm_1/embedding_35/embedding_lookup*^x_deep_fm_1/embedding_36/embedding_lookup*^x_deep_fm_1/embedding_37/embedding_lookup*^x_deep_fm_1/embedding_38/embedding_lookup*^x_deep_fm_1/embedding_39/embedding_lookup*^x_deep_fm_1/embedding_40/embedding_lookup*^x_deep_fm_1/embedding_41/embedding_lookup*^x_deep_fm_1/embedding_42/embedding_lookup*^x_deep_fm_1/embedding_43/embedding_lookup*^x_deep_fm_1/embedding_44/embedding_lookup*^x_deep_fm_1/embedding_45/embedding_lookup*^x_deep_fm_1/embedding_46/embedding_lookup*^x_deep_fm_1/embedding_47/embedding_lookup*^x_deep_fm_1/embedding_48/embedding_lookup*^x_deep_fm_1/embedding_49/embedding_lookup*^x_deep_fm_1/embedding_50/embedding_lookup*^x_deep_fm_1/embedding_51/embedding_lookup4^x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp3^x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp5x_deep_fm_1/batch_normalization_1/Cast/ReadVariableOp2r
7x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp7x_deep_fm_1/batch_normalization_1/Cast_1/ReadVariableOp2r
7x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp7x_deep_fm_1/batch_normalization_1/Cast_2/ReadVariableOp2r
7x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp7x_deep_fm_1/batch_normalization_1/Cast_3/ReadVariableOp2l
4x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp4x_deep_fm_1/cin_1/conv1d/ExpandDims_1/ReadVariableOp2p
6x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp6x_deep_fm_1/cin_1/conv1d_1/ExpandDims_1/ReadVariableOp2X
*x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp*x_deep_fm_1/dense_9/BiasAdd/ReadVariableOp2V
)x_deep_fm_1/dense_9/MatMul/ReadVariableOp)x_deep_fm_1/dense_9/MatMul/ReadVariableOp2t
8x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp8x_deep_fm_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp2r
7x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp7x_deep_fm_1/dense_layer_1/dense_6/MatMul/ReadVariableOp2t
8x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp8x_deep_fm_1/dense_layer_1/dense_7/BiasAdd/ReadVariableOp2r
7x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp7x_deep_fm_1/dense_layer_1/dense_7/MatMul/ReadVariableOp2t
8x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp8x_deep_fm_1/dense_layer_1/dense_8/BiasAdd/ReadVariableOp2r
7x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp7x_deep_fm_1/dense_layer_1/dense_8/MatMul/ReadVariableOp2V
)x_deep_fm_1/embedding_26/embedding_lookup)x_deep_fm_1/embedding_26/embedding_lookup2V
)x_deep_fm_1/embedding_27/embedding_lookup)x_deep_fm_1/embedding_27/embedding_lookup2V
)x_deep_fm_1/embedding_28/embedding_lookup)x_deep_fm_1/embedding_28/embedding_lookup2V
)x_deep_fm_1/embedding_29/embedding_lookup)x_deep_fm_1/embedding_29/embedding_lookup2V
)x_deep_fm_1/embedding_30/embedding_lookup)x_deep_fm_1/embedding_30/embedding_lookup2V
)x_deep_fm_1/embedding_31/embedding_lookup)x_deep_fm_1/embedding_31/embedding_lookup2V
)x_deep_fm_1/embedding_32/embedding_lookup)x_deep_fm_1/embedding_32/embedding_lookup2V
)x_deep_fm_1/embedding_33/embedding_lookup)x_deep_fm_1/embedding_33/embedding_lookup2V
)x_deep_fm_1/embedding_34/embedding_lookup)x_deep_fm_1/embedding_34/embedding_lookup2V
)x_deep_fm_1/embedding_35/embedding_lookup)x_deep_fm_1/embedding_35/embedding_lookup2V
)x_deep_fm_1/embedding_36/embedding_lookup)x_deep_fm_1/embedding_36/embedding_lookup2V
)x_deep_fm_1/embedding_37/embedding_lookup)x_deep_fm_1/embedding_37/embedding_lookup2V
)x_deep_fm_1/embedding_38/embedding_lookup)x_deep_fm_1/embedding_38/embedding_lookup2V
)x_deep_fm_1/embedding_39/embedding_lookup)x_deep_fm_1/embedding_39/embedding_lookup2V
)x_deep_fm_1/embedding_40/embedding_lookup)x_deep_fm_1/embedding_40/embedding_lookup2V
)x_deep_fm_1/embedding_41/embedding_lookup)x_deep_fm_1/embedding_41/embedding_lookup2V
)x_deep_fm_1/embedding_42/embedding_lookup)x_deep_fm_1/embedding_42/embedding_lookup2V
)x_deep_fm_1/embedding_43/embedding_lookup)x_deep_fm_1/embedding_43/embedding_lookup2V
)x_deep_fm_1/embedding_44/embedding_lookup)x_deep_fm_1/embedding_44/embedding_lookup2V
)x_deep_fm_1/embedding_45/embedding_lookup)x_deep_fm_1/embedding_45/embedding_lookup2V
)x_deep_fm_1/embedding_46/embedding_lookup)x_deep_fm_1/embedding_46/embedding_lookup2V
)x_deep_fm_1/embedding_47/embedding_lookup)x_deep_fm_1/embedding_47/embedding_lookup2V
)x_deep_fm_1/embedding_48/embedding_lookup)x_deep_fm_1/embedding_48/embedding_lookup2V
)x_deep_fm_1/embedding_49/embedding_lookup)x_deep_fm_1/embedding_49/embedding_lookup2V
)x_deep_fm_1/embedding_50/embedding_lookup)x_deep_fm_1/embedding_50/embedding_lookup2V
)x_deep_fm_1/embedding_51/embedding_lookup)x_deep_fm_1/embedding_51/embedding_lookup2j
3x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp3x_deep_fm_1/linear_1/dense_5/BiasAdd/ReadVariableOp2h
2x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp2x_deep_fm_1/linear_1/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
?

?
G__inference_embedding_31_layer_call_and_return_conditional_losses_89399

inputs(
embedding_lookup_89393:	
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89393Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89393*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89393*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_28_layer_call_and_return_conditional_losses_85411

inputs)
embedding_lookup_85405:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85405Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85405*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85405*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_40_layer_call_and_return_conditional_losses_89552

inputs)
embedding_lookup_89546:	?

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89546Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89546*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89546*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_36_layer_call_fn_89474

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_855552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_29_layer_call_and_return_conditional_losses_89365

inputs)
embedding_lookup_89359:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89359Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89359*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89359*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_43_layer_call_and_return_conditional_losses_89603

inputs)
embedding_lookup_89597:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89597Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89597*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89597*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_44_layer_call_and_return_conditional_losses_89620

inputs)
embedding_lookup_89614:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89614Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89614*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89614*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_40_layer_call_fn_89542

inputs
unknown:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_856272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_47_layer_call_fn_89661

inputs
unknown:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_857532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_27_layer_call_and_return_conditional_losses_89331

inputs)
embedding_lookup_89325:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89325Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89325*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89325*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_26_layer_call_fn_89304

inputs
unknown:h
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_853752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_36_layer_call_and_return_conditional_losses_85555

inputs)
embedding_lookup_85549:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85549Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85549*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85549*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_41_layer_call_fn_89559

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_856452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
?
@__inference_cin_1_layer_call_and_return_conditional_losses_89139

inputsB
+conv1d_expanddims_1_readvariableop_resource:? D
-conv1d_1_expanddims_1_readvariableop_resource:? 
identity??"conv1d/ExpandDims_1/ReadVariableOp?$conv1d_1/ExpandDims_1/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
splitq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2	
split_1?
MatMul/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/a?
MatMul/bPacksplit_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/b?
MatMulBatchMatMulV2MatMul/a:output:0MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
MatMuls
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
Reshape/shape}
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapeu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposey
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimstranspose:y:0conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeezey
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transposeconv1d/Squeeze:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1q
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2	
split_2?

MatMul_1/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul_1/a?

MatMul_1/bPacksplit_2:output:0split_2:output:1split_2:output:2split_2:output:3split_2:output:4split_2:output:5split_2:output:6split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2

MatMul_1/b?
MatMul_1BatchMatMulV2MatMul_1/a:output:0MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2

MatMul_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
Reshape_1/shape?
	Reshape_1ReshapeMatMul_1:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
	Reshape_1y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_1:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2}
conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_1/ExpandDims/dim?
conv1d_1/ExpandDims
ExpandDimstranspose_2:y:0 conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_1/ExpandDims?
$conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02&
$conv1d_1/ExpandDims_1/ReadVariableOpx
conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_1/ExpandDims_1/dim?
conv1d_1/ExpandDims_1
ExpandDims,conv1d_1/ExpandDims_1/ReadVariableOp:value:0"conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d_1/ExpandDims_1?
conv1d_1Conv2Dconv1d_1/ExpandDims:output:0conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2

conv1d_1?
conv1d_1/SqueezeSqueezeconv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/Squeezey
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm?
transpose_3	Transposeconv1d_1/Squeeze:output:0transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2transpose_1:y:0transpose_3:y:0concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicest
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1g
IdentityIdentitySum:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp%^conv1d_1/ExpandDims_1/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2L
$conv1d_1/ExpandDims_1/ReadVariableOp$conv1d_1/ExpandDims_1/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_linear_1_layer_call_and_return_conditional_losses_88899

inputs8
&dense_5_matmul_readvariableop_resource:'5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
??
?"
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88352

inputs@
2batch_normalization_1_cast_readvariableop_resource:'B
4batch_normalization_1_cast_1_readvariableop_resource:'B
4batch_normalization_1_cast_2_readvariableop_resource:'B
4batch_normalization_1_cast_3_readvariableop_resource:'A
/linear_1_dense_5_matmul_readvariableop_resource:'>
0linear_1_dense_5_biasadd_readvariableop_resource:5
#embedding_26_embedding_lookup_87964:h6
#embedding_27_embedding_lookup_87974:	?6
#embedding_28_embedding_lookup_87984:	?6
#embedding_29_embedding_lookup_87994:	?5
#embedding_30_embedding_lookup_88004:*5
#embedding_31_embedding_lookup_88014:	6
#embedding_32_embedding_lookup_88024:	?5
#embedding_33_embedding_lookup_88034:C5
#embedding_34_embedding_lookup_88044:6
#embedding_35_embedding_lookup_88054:	?6
#embedding_36_embedding_lookup_88064:	?6
#embedding_37_embedding_lookup_88074:	?6
#embedding_38_embedding_lookup_88084:	?5
#embedding_39_embedding_lookup_88094:6
#embedding_40_embedding_lookup_88104:	?
6
#embedding_41_embedding_lookup_88114:	?5
#embedding_42_embedding_lookup_88124:
6
#embedding_43_embedding_lookup_88134:	?6
#embedding_44_embedding_lookup_88144:	?5
#embedding_45_embedding_lookup_88154:6
#embedding_46_embedding_lookup_88164:	?5
#embedding_47_embedding_lookup_88174:
5
#embedding_48_embedding_lookup_88184:6
#embedding_49_embedding_lookup_88194:	?5
#embedding_50_embedding_lookup_88204:'6
#embedding_51_embedding_lookup_88214:	?H
1cin_1_conv1d_expanddims_1_readvariableop_resource:? J
3cin_1_conv1d_1_expanddims_1_readvariableop_resource:? G
4dense_layer_1_dense_6_matmul_readvariableop_resource:	?@C
5dense_layer_1_dense_6_biasadd_readvariableop_resource:@F
4dense_layer_1_dense_7_matmul_readvariableop_resource:@@C
5dense_layer_1_dense_7_biasadd_readvariableop_resource:@F
4dense_layer_1_dense_8_matmul_readvariableop_resource:@C
5dense_layer_1_dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:@5
'dense_9_biasadd_readvariableop_resource:
identity??)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?+batch_normalization_1/Cast_2/ReadVariableOp?+batch_normalization_1/Cast_3/ReadVariableOp?(cin_1/conv1d/ExpandDims_1/ReadVariableOp?*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?,dense_layer_1/dense_6/BiasAdd/ReadVariableOp?+dense_layer_1/dense_6/MatMul/ReadVariableOp?,dense_layer_1/dense_7/BiasAdd/ReadVariableOp?+dense_layer_1/dense_7/MatMul/ReadVariableOp?,dense_layer_1/dense_8/BiasAdd/ReadVariableOp?+dense_layer_1/dense_8/MatMul/ReadVariableOp?embedding_26/embedding_lookup?embedding_27/embedding_lookup?embedding_28/embedding_lookup?embedding_29/embedding_lookup?embedding_30/embedding_lookup?embedding_31/embedding_lookup?embedding_32/embedding_lookup?embedding_33/embedding_lookup?embedding_34/embedding_lookup?embedding_35/embedding_lookup?embedding_36/embedding_lookup?embedding_37/embedding_lookup?embedding_38/embedding_lookup?embedding_39/embedding_lookup?embedding_40/embedding_lookup?embedding_41/embedding_lookup?embedding_42/embedding_lookup?embedding_43/embedding_lookup?embedding_44/embedding_lookup?embedding_45/embedding_lookup?embedding_46/embedding_lookup?embedding_47/embedding_lookup?embedding_48/embedding_lookup?embedding_49/embedding_lookup?embedding_50/embedding_lookup?embedding_51/embedding_lookup?'linear_1/dense_5/BiasAdd/ReadVariableOp?&linear_1/dense_5/MatMul/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:'*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:'*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
:'*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp?
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
:'*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:'2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulinputs'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:'2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2'
%batch_normalization_1/batchnorm/add_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice)batch_normalization_1/batchnorm/add_1:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice)batch_normalization_1/batchnorm/add_1:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
&linear_1/dense_5/MatMul/ReadVariableOpReadVariableOp/linear_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02(
&linear_1/dense_5/MatMul/ReadVariableOp?
linear_1/dense_5/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0.linear_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
linear_1/dense_5/MatMul?
'linear_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp0linear_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'linear_1/dense_5/BiasAdd/ReadVariableOp?
linear_1/dense_5/BiasAddBiasAdd!linear_1/dense_5/MatMul:product:0/linear_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
linear_1/dense_5/BiasAdd
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
embedding_26/CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_26/Cast?
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_87964embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/87964*'
_output_shapes
:?????????*
dtype02
embedding_26/embedding_lookup?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/87964*'
_output_shapes
:?????????2(
&embedding_26/embedding_lookup/Identity?
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_26/embedding_lookup/Identity_1
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
embedding_27/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_27/Cast?
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_87974embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/87974*'
_output_shapes
:?????????*
dtype02
embedding_27/embedding_lookup?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/87974*'
_output_shapes
:?????????2(
&embedding_27/embedding_lookup/Identity?
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_27/embedding_lookup/Identity_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
embedding_28/CastCaststrided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_28/Cast?
embedding_28/embedding_lookupResourceGather#embedding_28_embedding_lookup_87984embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_28/embedding_lookup/87984*'
_output_shapes
:?????????*
dtype02
embedding_28/embedding_lookup?
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_28/embedding_lookup/87984*'
_output_shapes
:?????????2(
&embedding_28/embedding_lookup/Identity?
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_28/embedding_lookup/Identity_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
embedding_29/CastCaststrided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_29/Cast?
embedding_29/embedding_lookupResourceGather#embedding_29_embedding_lookup_87994embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_29/embedding_lookup/87994*'
_output_shapes
:?????????*
dtype02
embedding_29/embedding_lookup?
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_29/embedding_lookup/87994*'
_output_shapes
:?????????2(
&embedding_29/embedding_lookup/Identity?
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_29/embedding_lookup/Identity_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
embedding_30/CastCaststrided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather#embedding_30_embedding_lookup_88004embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_30/embedding_lookup/88004*'
_output_shapes
:?????????*
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_30/embedding_lookup/88004*'
_output_shapes
:?????????2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_30/embedding_lookup/Identity_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
embedding_31/CastCaststrided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_31/Cast?
embedding_31/embedding_lookupResourceGather#embedding_31_embedding_lookup_88014embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_31/embedding_lookup/88014*'
_output_shapes
:?????????*
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_31/embedding_lookup/88014*'
_output_shapes
:?????????2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_31/embedding_lookup/Identity_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
embedding_32/CastCaststrided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_32/Cast?
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_88024embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/88024*'
_output_shapes
:?????????*
dtype02
embedding_32/embedding_lookup?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/88024*'
_output_shapes
:?????????2(
&embedding_32/embedding_lookup/Identity?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_32/embedding_lookup/Identity_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
embedding_33/CastCaststrided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_33/Cast?
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_88034embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/88034*'
_output_shapes
:?????????*
dtype02
embedding_33/embedding_lookup?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/88034*'
_output_shapes
:?????????2(
&embedding_33/embedding_lookup/Identity?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_33/embedding_lookup/Identity_1?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
embedding_34/CastCaststrided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_34/Cast?
embedding_34/embedding_lookupResourceGather#embedding_34_embedding_lookup_88044embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_34/embedding_lookup/88044*'
_output_shapes
:?????????*
dtype02
embedding_34/embedding_lookup?
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_34/embedding_lookup/88044*'
_output_shapes
:?????????2(
&embedding_34/embedding_lookup/Identity?
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_34/embedding_lookup/Identity_1?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
embedding_35/CastCaststrided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_35/Cast?
embedding_35/embedding_lookupResourceGather#embedding_35_embedding_lookup_88054embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_35/embedding_lookup/88054*'
_output_shapes
:?????????*
dtype02
embedding_35/embedding_lookup?
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_35/embedding_lookup/88054*'
_output_shapes
:?????????2(
&embedding_35/embedding_lookup/Identity?
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_35/embedding_lookup/Identity_1?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
embedding_36/CastCaststrided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_36/Cast?
embedding_36/embedding_lookupResourceGather#embedding_36_embedding_lookup_88064embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_36/embedding_lookup/88064*'
_output_shapes
:?????????*
dtype02
embedding_36/embedding_lookup?
&embedding_36/embedding_lookup/IdentityIdentity&embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_36/embedding_lookup/88064*'
_output_shapes
:?????????2(
&embedding_36/embedding_lookup/Identity?
(embedding_36/embedding_lookup/Identity_1Identity/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_36/embedding_lookup/Identity_1?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
embedding_37/CastCaststrided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_37/Cast?
embedding_37/embedding_lookupResourceGather#embedding_37_embedding_lookup_88074embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_37/embedding_lookup/88074*'
_output_shapes
:?????????*
dtype02
embedding_37/embedding_lookup?
&embedding_37/embedding_lookup/IdentityIdentity&embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_37/embedding_lookup/88074*'
_output_shapes
:?????????2(
&embedding_37/embedding_lookup/Identity?
(embedding_37/embedding_lookup/Identity_1Identity/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_37/embedding_lookup/Identity_1?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
embedding_38/CastCaststrided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_38/Cast?
embedding_38/embedding_lookupResourceGather#embedding_38_embedding_lookup_88084embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_38/embedding_lookup/88084*'
_output_shapes
:?????????*
dtype02
embedding_38/embedding_lookup?
&embedding_38/embedding_lookup/IdentityIdentity&embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_38/embedding_lookup/88084*'
_output_shapes
:?????????2(
&embedding_38/embedding_lookup/Identity?
(embedding_38/embedding_lookup/Identity_1Identity/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_38/embedding_lookup/Identity_1?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
embedding_39/CastCaststrided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_39/Cast?
embedding_39/embedding_lookupResourceGather#embedding_39_embedding_lookup_88094embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_39/embedding_lookup/88094*'
_output_shapes
:?????????*
dtype02
embedding_39/embedding_lookup?
&embedding_39/embedding_lookup/IdentityIdentity&embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_39/embedding_lookup/88094*'
_output_shapes
:?????????2(
&embedding_39/embedding_lookup/Identity?
(embedding_39/embedding_lookup/Identity_1Identity/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_39/embedding_lookup/Identity_1?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
embedding_40/CastCaststrided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_40/Cast?
embedding_40/embedding_lookupResourceGather#embedding_40_embedding_lookup_88104embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_40/embedding_lookup/88104*'
_output_shapes
:?????????*
dtype02
embedding_40/embedding_lookup?
&embedding_40/embedding_lookup/IdentityIdentity&embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_40/embedding_lookup/88104*'
_output_shapes
:?????????2(
&embedding_40/embedding_lookup/Identity?
(embedding_40/embedding_lookup/Identity_1Identity/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_40/embedding_lookup/Identity_1?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
embedding_41/CastCaststrided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_41/Cast?
embedding_41/embedding_lookupResourceGather#embedding_41_embedding_lookup_88114embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_41/embedding_lookup/88114*'
_output_shapes
:?????????*
dtype02
embedding_41/embedding_lookup?
&embedding_41/embedding_lookup/IdentityIdentity&embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_41/embedding_lookup/88114*'
_output_shapes
:?????????2(
&embedding_41/embedding_lookup/Identity?
(embedding_41/embedding_lookup/Identity_1Identity/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_41/embedding_lookup/Identity_1?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
embedding_42/CastCaststrided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_42/Cast?
embedding_42/embedding_lookupResourceGather#embedding_42_embedding_lookup_88124embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_42/embedding_lookup/88124*'
_output_shapes
:?????????*
dtype02
embedding_42/embedding_lookup?
&embedding_42/embedding_lookup/IdentityIdentity&embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_42/embedding_lookup/88124*'
_output_shapes
:?????????2(
&embedding_42/embedding_lookup/Identity?
(embedding_42/embedding_lookup/Identity_1Identity/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_42/embedding_lookup/Identity_1?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
embedding_43/CastCaststrided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_43/Cast?
embedding_43/embedding_lookupResourceGather#embedding_43_embedding_lookup_88134embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_43/embedding_lookup/88134*'
_output_shapes
:?????????*
dtype02
embedding_43/embedding_lookup?
&embedding_43/embedding_lookup/IdentityIdentity&embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_43/embedding_lookup/88134*'
_output_shapes
:?????????2(
&embedding_43/embedding_lookup/Identity?
(embedding_43/embedding_lookup/Identity_1Identity/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_43/embedding_lookup/Identity_1?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
embedding_44/CastCaststrided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_44/Cast?
embedding_44/embedding_lookupResourceGather#embedding_44_embedding_lookup_88144embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_44/embedding_lookup/88144*'
_output_shapes
:?????????*
dtype02
embedding_44/embedding_lookup?
&embedding_44/embedding_lookup/IdentityIdentity&embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_44/embedding_lookup/88144*'
_output_shapes
:?????????2(
&embedding_44/embedding_lookup/Identity?
(embedding_44/embedding_lookup/Identity_1Identity/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_44/embedding_lookup/Identity_1?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
embedding_45/CastCaststrided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_45/Cast?
embedding_45/embedding_lookupResourceGather#embedding_45_embedding_lookup_88154embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_45/embedding_lookup/88154*'
_output_shapes
:?????????*
dtype02
embedding_45/embedding_lookup?
&embedding_45/embedding_lookup/IdentityIdentity&embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_45/embedding_lookup/88154*'
_output_shapes
:?????????2(
&embedding_45/embedding_lookup/Identity?
(embedding_45/embedding_lookup/Identity_1Identity/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_45/embedding_lookup/Identity_1?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
embedding_46/CastCaststrided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_46/Cast?
embedding_46/embedding_lookupResourceGather#embedding_46_embedding_lookup_88164embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_46/embedding_lookup/88164*'
_output_shapes
:?????????*
dtype02
embedding_46/embedding_lookup?
&embedding_46/embedding_lookup/IdentityIdentity&embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_46/embedding_lookup/88164*'
_output_shapes
:?????????2(
&embedding_46/embedding_lookup/Identity?
(embedding_46/embedding_lookup/Identity_1Identity/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_46/embedding_lookup/Identity_1?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
embedding_47/CastCaststrided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_47/Cast?
embedding_47/embedding_lookupResourceGather#embedding_47_embedding_lookup_88174embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_47/embedding_lookup/88174*'
_output_shapes
:?????????*
dtype02
embedding_47/embedding_lookup?
&embedding_47/embedding_lookup/IdentityIdentity&embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_47/embedding_lookup/88174*'
_output_shapes
:?????????2(
&embedding_47/embedding_lookup/Identity?
(embedding_47/embedding_lookup/Identity_1Identity/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_47/embedding_lookup/Identity_1?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
embedding_48/CastCaststrided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_48/Cast?
embedding_48/embedding_lookupResourceGather#embedding_48_embedding_lookup_88184embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_48/embedding_lookup/88184*'
_output_shapes
:?????????*
dtype02
embedding_48/embedding_lookup?
&embedding_48/embedding_lookup/IdentityIdentity&embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_48/embedding_lookup/88184*'
_output_shapes
:?????????2(
&embedding_48/embedding_lookup/Identity?
(embedding_48/embedding_lookup/Identity_1Identity/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_48/embedding_lookup/Identity_1?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
embedding_49/CastCaststrided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_49/Cast?
embedding_49/embedding_lookupResourceGather#embedding_49_embedding_lookup_88194embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_49/embedding_lookup/88194*'
_output_shapes
:?????????*
dtype02
embedding_49/embedding_lookup?
&embedding_49/embedding_lookup/IdentityIdentity&embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_49/embedding_lookup/88194*'
_output_shapes
:?????????2(
&embedding_49/embedding_lookup/Identity?
(embedding_49/embedding_lookup/Identity_1Identity/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_49/embedding_lookup/Identity_1?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
embedding_50/CastCaststrided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_50/Cast?
embedding_50/embedding_lookupResourceGather#embedding_50_embedding_lookup_88204embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_50/embedding_lookup/88204*'
_output_shapes
:?????????*
dtype02
embedding_50/embedding_lookup?
&embedding_50/embedding_lookup/IdentityIdentity&embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_50/embedding_lookup/88204*'
_output_shapes
:?????????2(
&embedding_50/embedding_lookup/Identity?
(embedding_50/embedding_lookup/Identity_1Identity/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_50/embedding_lookup/Identity_1?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
embedding_51/CastCaststrided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_51/Cast?
embedding_51/embedding_lookupResourceGather#embedding_51_embedding_lookup_88214embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_51/embedding_lookup/88214*'
_output_shapes
:?????????*
dtype02
embedding_51/embedding_lookup?
&embedding_51/embedding_lookup/IdentityIdentity&embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_51/embedding_lookup/88214*'
_output_shapes
:?????????2(
&embedding_51/embedding_lookup/Identity?
(embedding_51/embedding_lookup/Identity_1Identity/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_51/embedding_lookup/Identity_1?
packedPack1embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:01embedding_28/embedding_lookup/Identity_1:output:01embedding_29/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:01embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:01embedding_34/embedding_lookup/Identity_1:output:01embedding_35/embedding_lookup/Identity_1:output:01embedding_36/embedding_lookup/Identity_1:output:01embedding_37/embedding_lookup/Identity_1:output:01embedding_38/embedding_lookup/Identity_1:output:01embedding_39/embedding_lookup/Identity_1:output:01embedding_40/embedding_lookup/Identity_1:output:01embedding_41/embedding_lookup/Identity_1:output:01embedding_42/embedding_lookup/Identity_1:output:01embedding_43/embedding_lookup/Identity_1:output:01embedding_44/embedding_lookup/Identity_1:output:01embedding_45/embedding_lookup/Identity_1:output:01embedding_46/embedding_lookup/Identity_1:output:01embedding_47/embedding_lookup/Identity_1:output:01embedding_48/embedding_lookup/Identity_1:output:01embedding_49/embedding_lookup/Identity_1:output:01embedding_50/embedding_lookup/Identity_1:output:01embedding_51/embedding_lookup/Identity_1:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposey
cin_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split/split_dim?
cin_1/splitSplitcin_1/split/split_dim:output:0transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
cin_1/split}
cin_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split_1/split_dim?
cin_1/split_1Split cin_1/split_1/split_dim:output:0transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
cin_1/split_1?
cin_1/MatMul/aPackcin_1/split:output:0cin_1/split:output:1cin_1/split:output:2cin_1/split:output:3cin_1/split:output:4cin_1/split:output:5cin_1/split:output:6cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul/a?
cin_1/MatMul/bPackcin_1/split_1:output:0cin_1/split_1:output:1cin_1/split_1:output:2cin_1/split_1:output:3cin_1/split_1:output:4cin_1/split_1:output:5cin_1/split_1:output:6cin_1/split_1:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul/b?
cin_1/MatMulBatchMatMulV2cin_1/MatMul/a:output:0cin_1/MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
cin_1/MatMul
cin_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
cin_1/Reshape/shape?
cin_1/ReshapeReshapecin_1/MatMul:output:0cin_1/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
cin_1/Reshape?
cin_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose/perm?
cin_1/transpose	Transposecin_1/Reshape:output:0cin_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
cin_1/transpose?
cin_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/conv1d/ExpandDims/dim?
cin_1/conv1d/ExpandDims
ExpandDimscin_1/transpose:y:0$cin_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
cin_1/conv1d/ExpandDims?
(cin_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02*
(cin_1/conv1d/ExpandDims_1/ReadVariableOp?
cin_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
cin_1/conv1d/ExpandDims_1/dim?
cin_1/conv1d/ExpandDims_1
ExpandDims0cin_1/conv1d/ExpandDims_1/ReadVariableOp:value:0&cin_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
cin_1/conv1d/ExpandDims_1?
cin_1/conv1dConv2D cin_1/conv1d/ExpandDims:output:0"cin_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
cin_1/conv1d?
cin_1/conv1d/SqueezeSqueezecin_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
cin_1/conv1d/Squeeze?
cin_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_1/perm?
cin_1/transpose_1	Transposecin_1/conv1d/Squeeze:output:0cin_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
cin_1/transpose_1}
cin_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split_2/split_dim?
cin_1/split_2Split cin_1/split_2/split_dim:output:0cin_1/transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2
cin_1/split_2?
cin_1/MatMul_1/aPackcin_1/split:output:0cin_1/split:output:1cin_1/split:output:2cin_1/split:output:3cin_1/split:output:4cin_1/split:output:5cin_1/split:output:6cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul_1/a?
cin_1/MatMul_1/bPackcin_1/split_2:output:0cin_1/split_2:output:1cin_1/split_2:output:2cin_1/split_2:output:3cin_1/split_2:output:4cin_1/split_2:output:5cin_1/split_2:output:6cin_1/split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2
cin_1/MatMul_1/b?
cin_1/MatMul_1BatchMatMulV2cin_1/MatMul_1/a:output:0cin_1/MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2
cin_1/MatMul_1?
cin_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
cin_1/Reshape_1/shape?
cin_1/Reshape_1Reshapecin_1/MatMul_1:output:0cin_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
cin_1/Reshape_1?
cin_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_2/perm?
cin_1/transpose_2	Transposecin_1/Reshape_1:output:0cin_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
cin_1/transpose_2?
cin_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/conv1d_1/ExpandDims/dim?
cin_1/conv1d_1/ExpandDims
ExpandDimscin_1/transpose_2:y:0&cin_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
cin_1/conv1d_1/ExpandDims?
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02,
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?
cin_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
cin_1/conv1d_1/ExpandDims_1/dim?
cin_1/conv1d_1/ExpandDims_1
ExpandDims2cin_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0(cin_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
cin_1/conv1d_1/ExpandDims_1?
cin_1/conv1d_1Conv2D"cin_1/conv1d_1/ExpandDims:output:0$cin_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
cin_1/conv1d_1?
cin_1/conv1d_1/SqueezeSqueezecin_1/conv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
cin_1/conv1d_1/Squeeze?
cin_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_3/perm?
cin_1/transpose_3	Transposecin_1/conv1d_1/Squeeze:output:0cin_1/transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
cin_1/transpose_3h
cin_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
cin_1/concat/axis?
cin_1/concatConcatV2cin_1/transpose_1:y:0cin_1/transpose_3:y:0cin_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
cin_1/concat?
cin_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/Sum/reduction_indices?
	cin_1/SumSumcin_1/concat:output:0$cin_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
	cin_1/Sumo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
+dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+dense_layer_1/dense_6/MatMul/ReadVariableOp?
dense_layer_1/dense_6/MatMulMatMulconcat:output:03dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/MatMul?
,dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp?
dense_layer_1/dense_6/BiasAddBiasAdd&dense_layer_1/dense_6/MatMul:product:04dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/BiasAdd?
dense_layer_1/dense_6/ReluRelu&dense_layer_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/Relu?
+dense_layer_1/dense_7/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02-
+dense_layer_1/dense_7/MatMul/ReadVariableOp?
dense_layer_1/dense_7/MatMulMatMul(dense_layer_1/dense_6/Relu:activations:03dense_layer_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/MatMul?
,dense_layer_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dense_layer_1/dense_7/BiasAdd/ReadVariableOp?
dense_layer_1/dense_7/BiasAddBiasAdd&dense_layer_1/dense_7/MatMul:product:04dense_layer_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/BiasAdd?
dense_layer_1/dense_7/ReluRelu&dense_layer_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/Relu?
 dense_layer_1/dropout_1/IdentityIdentity(dense_layer_1/dense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2"
 dense_layer_1/dropout_1/Identity?
+dense_layer_1/dense_8/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02-
+dense_layer_1/dense_8/MatMul/ReadVariableOp?
dense_layer_1/dense_8/MatMulMatMul)dense_layer_1/dropout_1/Identity:output:03dense_layer_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_1/dense_8/MatMul?
,dense_layer_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_layer_1/dense_8/BiasAdd/ReadVariableOp?
dense_layer_1/dense_8/BiasAddBiasAdd&dense_layer_1/dense_8/MatMul:product:04dense_layer_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_1/dense_8/BiasAdd|
addAddV2!linear_1/dense_5/BiasAdd:output:0cin_1/Sum:output:0*
T0*'
_output_shapes
:?????????@2
addz
add_1AddV2add:z:0&dense_layer_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul	add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddi
SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp)^cin_1/conv1d/ExpandDims_1/ReadVariableOp+^cin_1/conv1d_1/ExpandDims_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp-^dense_layer_1/dense_6/BiasAdd/ReadVariableOp,^dense_layer_1/dense_6/MatMul/ReadVariableOp-^dense_layer_1/dense_7/BiasAdd/ReadVariableOp,^dense_layer_1/dense_7/MatMul/ReadVariableOp-^dense_layer_1/dense_8/BiasAdd/ReadVariableOp,^dense_layer_1/dense_8/MatMul/ReadVariableOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup^embedding_36/embedding_lookup^embedding_37/embedding_lookup^embedding_38/embedding_lookup^embedding_39/embedding_lookup^embedding_40/embedding_lookup^embedding_41/embedding_lookup^embedding_42/embedding_lookup^embedding_43/embedding_lookup^embedding_44/embedding_lookup^embedding_45/embedding_lookup^embedding_46/embedding_lookup^embedding_47/embedding_lookup^embedding_48/embedding_lookup^embedding_49/embedding_lookup^embedding_50/embedding_lookup^embedding_51/embedding_lookup(^linear_1/dense_5/BiasAdd/ReadVariableOp'^linear_1/dense_5/MatMul/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2T
(cin_1/conv1d/ExpandDims_1/ReadVariableOp(cin_1/conv1d/ExpandDims_1/ReadVariableOp2X
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2\
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp,dense_layer_1/dense_6/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_6/MatMul/ReadVariableOp+dense_layer_1/dense_6/MatMul/ReadVariableOp2\
,dense_layer_1/dense_7/BiasAdd/ReadVariableOp,dense_layer_1/dense_7/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_7/MatMul/ReadVariableOp+dense_layer_1/dense_7/MatMul/ReadVariableOp2\
,dense_layer_1/dense_8/BiasAdd/ReadVariableOp,dense_layer_1/dense_8/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_8/MatMul/ReadVariableOp+dense_layer_1/dense_8/MatMul/ReadVariableOp2>
embedding_26/embedding_lookupembedding_26/embedding_lookup2>
embedding_27/embedding_lookupembedding_27/embedding_lookup2>
embedding_28/embedding_lookupembedding_28/embedding_lookup2>
embedding_29/embedding_lookupembedding_29/embedding_lookup2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup2>
embedding_32/embedding_lookupembedding_32/embedding_lookup2>
embedding_33/embedding_lookupembedding_33/embedding_lookup2>
embedding_34/embedding_lookupembedding_34/embedding_lookup2>
embedding_35/embedding_lookupembedding_35/embedding_lookup2>
embedding_36/embedding_lookupembedding_36/embedding_lookup2>
embedding_37/embedding_lookupembedding_37/embedding_lookup2>
embedding_38/embedding_lookupembedding_38/embedding_lookup2>
embedding_39/embedding_lookupembedding_39/embedding_lookup2>
embedding_40/embedding_lookupembedding_40/embedding_lookup2>
embedding_41/embedding_lookupembedding_41/embedding_lookup2>
embedding_42/embedding_lookupembedding_42/embedding_lookup2>
embedding_43/embedding_lookupembedding_43/embedding_lookup2>
embedding_44/embedding_lookupembedding_44/embedding_lookup2>
embedding_45/embedding_lookupembedding_45/embedding_lookup2>
embedding_46/embedding_lookupembedding_46/embedding_lookup2>
embedding_47/embedding_lookupembedding_47/embedding_lookup2>
embedding_48/embedding_lookupembedding_48/embedding_lookup2>
embedding_49/embedding_lookupembedding_49/embedding_lookup2>
embedding_50/embedding_lookupembedding_50/embedding_lookup2>
embedding_51/embedding_lookupembedding_51/embedding_lookup2R
'linear_1/dense_5/BiasAdd/ReadVariableOp'linear_1/dense_5/BiasAdd/ReadVariableOp2P
&linear_1/dense_5/MatMul/ReadVariableOp&linear_1/dense_5/MatMul/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_43_layer_call_and_return_conditional_losses_85681

inputs)
embedding_lookup_85675:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85675Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85675*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85675*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87350
input_1)
batch_normalization_1_87082:')
batch_normalization_1_87084:')
batch_normalization_1_87086:')
batch_normalization_1_87088:' 
linear_1_87099:'
linear_1_87101:$
embedding_26_87108:h%
embedding_27_87115:	?%
embedding_28_87122:	?%
embedding_29_87129:	?$
embedding_30_87136:*$
embedding_31_87143:	%
embedding_32_87150:	?$
embedding_33_87157:C$
embedding_34_87164:%
embedding_35_87171:	?%
embedding_36_87178:	?%
embedding_37_87185:	?%
embedding_38_87192:	?$
embedding_39_87199:%
embedding_40_87206:	?
%
embedding_41_87213:	?$
embedding_42_87220:
%
embedding_43_87227:	?%
embedding_44_87234:	?$
embedding_45_87241:%
embedding_46_87248:	?$
embedding_47_87255:
$
embedding_48_87262:%
embedding_49_87269:	?$
embedding_50_87276:'%
embedding_51_87283:	?"
cin_1_87289:? "
cin_1_87291:? &
dense_layer_1_87298:	?@!
dense_layer_1_87300:@%
dense_layer_1_87302:@@!
dense_layer_1_87304:@%
dense_layer_1_87306:@!
dense_layer_1_87308:
dense_9_87313:@
dense_9_87315:
identity??-batch_normalization_1/StatefulPartitionedCall?cin_1/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?%dense_layer_1/StatefulPartitionedCall?$embedding_26/StatefulPartitionedCall?$embedding_27/StatefulPartitionedCall?$embedding_28/StatefulPartitionedCall?$embedding_29/StatefulPartitionedCall?$embedding_30/StatefulPartitionedCall?$embedding_31/StatefulPartitionedCall?$embedding_32/StatefulPartitionedCall?$embedding_33/StatefulPartitionedCall?$embedding_34/StatefulPartitionedCall?$embedding_35/StatefulPartitionedCall?$embedding_36/StatefulPartitionedCall?$embedding_37/StatefulPartitionedCall?$embedding_38/StatefulPartitionedCall?$embedding_39/StatefulPartitionedCall?$embedding_40/StatefulPartitionedCall?$embedding_41/StatefulPartitionedCall?$embedding_42/StatefulPartitionedCall?$embedding_43/StatefulPartitionedCall?$embedding_44/StatefulPartitionedCall?$embedding_45/StatefulPartitionedCall?$embedding_46/StatefulPartitionedCall?$embedding_47/StatefulPartitionedCall?$embedding_48/StatefulPartitionedCall?$embedding_49/StatefulPartitionedCall?$embedding_50/StatefulPartitionedCall?$embedding_51/StatefulPartitionedCall? linear_1/StatefulPartitionedCall?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_normalization_1_87082batch_normalization_1_87084batch_normalization_1_87086batch_normalization_1_87088*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????'*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_851832/
-batch_normalization_1/StatefulPartitionedCall{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice6batch_normalization_1/StatefulPartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
 linear_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0linear_1_87099linear_1_87101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_853552"
 linear_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_87108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_853752&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_87115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_853932&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_87122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_854112&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_87129*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_854292&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_87136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_854472&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_87143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_854652&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_87150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_854832&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_87157*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_855012&
$embedding_33/StatefulPartitionedCall?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_87164*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_855192&
$embedding_34/StatefulPartitionedCall?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_87171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_855372&
$embedding_35/StatefulPartitionedCall?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_87178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_855552&
$embedding_36/StatefulPartitionedCall?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_87185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_855732&
$embedding_37/StatefulPartitionedCall?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_87192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_855912&
$embedding_38/StatefulPartitionedCall?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_87199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_856092&
$embedding_39/StatefulPartitionedCall?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_87206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_856272&
$embedding_40/StatefulPartitionedCall?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_87213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_856452&
$embedding_41/StatefulPartitionedCall?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_87220*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_856632&
$embedding_42/StatefulPartitionedCall?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_87227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_856812&
$embedding_43/StatefulPartitionedCall?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_87234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_856992&
$embedding_44/StatefulPartitionedCall?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_87241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_857172&
$embedding_45/StatefulPartitionedCall?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_87248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_857352&
$embedding_46/StatefulPartitionedCall?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_87255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_857532&
$embedding_47/StatefulPartitionedCall?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_87262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_857712&
$embedding_48/StatefulPartitionedCall?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_87269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_857892&
$embedding_49/StatefulPartitionedCall?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_87276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_858072&
$embedding_50/StatefulPartitionedCall?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_87283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_858252&
$embedding_51/StatefulPartitionedCall?

packedPack-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transpose?
cin_1/StatefulPartitionedCallStatefulPartitionedCalltranspose:y:0cin_1_87289cin_1_87291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_cin_1_layer_call_and_return_conditional_losses_859312
cin_1/StatefulPartitionedCallo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0dense_layer_1_87298dense_layer_1_87300dense_layer_1_87302dense_layer_1_87304dense_layer_1_87306dense_layer_1_87308*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_859662'
%dense_layer_1/StatefulPartitionedCall?
addAddV2)linear_1/StatefulPartitionedCall:output:0&cin_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add?
add_1AddV2add:z:0.dense_layer_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/StatefulPartitionedCallStatefulPartitionedCall	add_1:z:0dense_9_87313dense_9_87315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_859922!
dense_9/StatefulPartitionedCally
SigmoidSigmoid(dense_9/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_87289*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_87289*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOpcin_1_87291*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOpcin_1_87291*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall^cin_1/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall!^linear_1/StatefulPartitionedCall4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2>
cin_1/StatefulPartitionedCallcin_1/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2L
$embedding_26/StatefulPartitionedCall$embedding_26/StatefulPartitionedCall2L
$embedding_27/StatefulPartitionedCall$embedding_27/StatefulPartitionedCall2L
$embedding_28/StatefulPartitionedCall$embedding_28/StatefulPartitionedCall2L
$embedding_29/StatefulPartitionedCall$embedding_29/StatefulPartitionedCall2L
$embedding_30/StatefulPartitionedCall$embedding_30/StatefulPartitionedCall2L
$embedding_31/StatefulPartitionedCall$embedding_31/StatefulPartitionedCall2L
$embedding_32/StatefulPartitionedCall$embedding_32/StatefulPartitionedCall2L
$embedding_33/StatefulPartitionedCall$embedding_33/StatefulPartitionedCall2L
$embedding_34/StatefulPartitionedCall$embedding_34/StatefulPartitionedCall2L
$embedding_35/StatefulPartitionedCall$embedding_35/StatefulPartitionedCall2L
$embedding_36/StatefulPartitionedCall$embedding_36/StatefulPartitionedCall2L
$embedding_37/StatefulPartitionedCall$embedding_37/StatefulPartitionedCall2L
$embedding_38/StatefulPartitionedCall$embedding_38/StatefulPartitionedCall2L
$embedding_39/StatefulPartitionedCall$embedding_39/StatefulPartitionedCall2L
$embedding_40/StatefulPartitionedCall$embedding_40/StatefulPartitionedCall2L
$embedding_41/StatefulPartitionedCall$embedding_41/StatefulPartitionedCall2L
$embedding_42/StatefulPartitionedCall$embedding_42/StatefulPartitionedCall2L
$embedding_43/StatefulPartitionedCall$embedding_43/StatefulPartitionedCall2L
$embedding_44/StatefulPartitionedCall$embedding_44/StatefulPartitionedCall2L
$embedding_45/StatefulPartitionedCall$embedding_45/StatefulPartitionedCall2L
$embedding_46/StatefulPartitionedCall$embedding_46/StatefulPartitionedCall2L
$embedding_47/StatefulPartitionedCall$embedding_47/StatefulPartitionedCall2L
$embedding_48/StatefulPartitionedCall$embedding_48/StatefulPartitionedCall2L
$embedding_49/StatefulPartitionedCall$embedding_49/StatefulPartitionedCall2L
$embedding_50/StatefulPartitionedCall$embedding_50/StatefulPartitionedCall2L
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall2D
 linear_1/StatefulPartitionedCall linear_1/StatefulPartitionedCall2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
?
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_85966

inputs9
&dense_6_matmul_readvariableop_resource:	?@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
dropout_1/IdentityIdentitydense_7/Relu:activations:0*
T0*'
_output_shapes
:?????????@2
dropout_1/Identity?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldropout_1/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_33_layer_call_and_return_conditional_losses_85501

inputs(
embedding_lookup_85495:C
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85495Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85495*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85495*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_32_layer_call_fn_89406

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_854832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_31_layer_call_fn_89389

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_854652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_37_layer_call_and_return_conditional_losses_89501

inputs)
embedding_lookup_89495:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89495Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89495*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89495*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_50_layer_call_fn_89712

inputs
unknown:'
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_858072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88837

inputs*
cast_readvariableop_resource:',
cast_1_readvariableop_resource:',
cast_2_readvariableop_resource:',
cast_3_readvariableop_resource:'
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:'2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:'2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:'2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:'2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
C__inference_linear_1_layer_call_and_return_conditional_losses_88909

inputs8
&dense_5_matmul_readvariableop_resource:'5
'dense_5_biasadd_readvariableop_resource:
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
,__inference_embedding_35_layer_call_fn_89457

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_855372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_9_layer_call_and_return_conditional_losses_85992

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_embedding_42_layer_call_fn_89576

inputs
unknown:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_856632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_47_layer_call_and_return_conditional_losses_89671

inputs(
embedding_lookup_89665:

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89665Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89665*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89665*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_39_layer_call_and_return_conditional_losses_85609

inputs(
embedding_lookup_85603:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85603Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85603*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85603*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_38_layer_call_and_return_conditional_losses_85591

inputs)
embedding_lookup_85585:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85585Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85585*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85585*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?h
!__inference__traced_restore_90582
file_prefixF
8assignvariableop_x_deep_fm_1_batch_normalization_1_gamma:'G
9assignvariableop_1_x_deep_fm_1_batch_normalization_1_beta:'N
@assignvariableop_2_x_deep_fm_1_batch_normalization_1_moving_mean:'R
Dassignvariableop_3_x_deep_fm_1_batch_normalization_1_moving_variance:'>
'assignvariableop_4_x_deep_fm_1_cin_1_w0:? >
'assignvariableop_5_x_deep_fm_1_cin_1_w1:? ?
-assignvariableop_6_x_deep_fm_1_dense_9_kernel:@9
+assignvariableop_7_x_deep_fm_1_dense_9_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: I
7assignvariableop_13_x_deep_fm_1_embedding_26_embeddings:hJ
7assignvariableop_14_x_deep_fm_1_embedding_27_embeddings:	?J
7assignvariableop_15_x_deep_fm_1_embedding_28_embeddings:	?J
7assignvariableop_16_x_deep_fm_1_embedding_29_embeddings:	?I
7assignvariableop_17_x_deep_fm_1_embedding_30_embeddings:*I
7assignvariableop_18_x_deep_fm_1_embedding_31_embeddings:	J
7assignvariableop_19_x_deep_fm_1_embedding_32_embeddings:	?I
7assignvariableop_20_x_deep_fm_1_embedding_33_embeddings:CI
7assignvariableop_21_x_deep_fm_1_embedding_34_embeddings:J
7assignvariableop_22_x_deep_fm_1_embedding_35_embeddings:	?J
7assignvariableop_23_x_deep_fm_1_embedding_36_embeddings:	?J
7assignvariableop_24_x_deep_fm_1_embedding_37_embeddings:	?J
7assignvariableop_25_x_deep_fm_1_embedding_38_embeddings:	?I
7assignvariableop_26_x_deep_fm_1_embedding_39_embeddings:J
7assignvariableop_27_x_deep_fm_1_embedding_40_embeddings:	?
J
7assignvariableop_28_x_deep_fm_1_embedding_41_embeddings:	?I
7assignvariableop_29_x_deep_fm_1_embedding_42_embeddings:
J
7assignvariableop_30_x_deep_fm_1_embedding_43_embeddings:	?J
7assignvariableop_31_x_deep_fm_1_embedding_44_embeddings:	?I
7assignvariableop_32_x_deep_fm_1_embedding_45_embeddings:J
7assignvariableop_33_x_deep_fm_1_embedding_46_embeddings:	?I
7assignvariableop_34_x_deep_fm_1_embedding_47_embeddings:
I
7assignvariableop_35_x_deep_fm_1_embedding_48_embeddings:J
7assignvariableop_36_x_deep_fm_1_embedding_49_embeddings:	?I
7assignvariableop_37_x_deep_fm_1_embedding_50_embeddings:'J
7assignvariableop_38_x_deep_fm_1_embedding_51_embeddings:	?I
7assignvariableop_39_x_deep_fm_1_linear_1_dense_5_kernel:'C
5assignvariableop_40_x_deep_fm_1_linear_1_dense_5_bias:O
<assignvariableop_41_x_deep_fm_1_dense_layer_1_dense_6_kernel:	?@H
:assignvariableop_42_x_deep_fm_1_dense_layer_1_dense_6_bias:@N
<assignvariableop_43_x_deep_fm_1_dense_layer_1_dense_7_kernel:@@H
:assignvariableop_44_x_deep_fm_1_dense_layer_1_dense_7_bias:@N
<assignvariableop_45_x_deep_fm_1_dense_layer_1_dense_8_kernel:@H
:assignvariableop_46_x_deep_fm_1_dense_layer_1_dense_8_bias:#
assignvariableop_47_total: #
assignvariableop_48_count: 1
"assignvariableop_49_true_positives:	?1
"assignvariableop_50_true_negatives:	?2
#assignvariableop_51_false_positives:	?2
#assignvariableop_52_false_negatives:	?2
$assignvariableop_53_true_positives_1:3
%assignvariableop_54_false_negatives_1:P
Bassignvariableop_55_adam_x_deep_fm_1_batch_normalization_1_gamma_m:'O
Aassignvariableop_56_adam_x_deep_fm_1_batch_normalization_1_beta_m:'F
/assignvariableop_57_adam_x_deep_fm_1_cin_1_w0_m:? F
/assignvariableop_58_adam_x_deep_fm_1_cin_1_w1_m:? G
5assignvariableop_59_adam_x_deep_fm_1_dense_9_kernel_m:@A
3assignvariableop_60_adam_x_deep_fm_1_dense_9_bias_m:P
>assignvariableop_61_adam_x_deep_fm_1_embedding_26_embeddings_m:hQ
>assignvariableop_62_adam_x_deep_fm_1_embedding_27_embeddings_m:	?Q
>assignvariableop_63_adam_x_deep_fm_1_embedding_28_embeddings_m:	?Q
>assignvariableop_64_adam_x_deep_fm_1_embedding_29_embeddings_m:	?P
>assignvariableop_65_adam_x_deep_fm_1_embedding_30_embeddings_m:*P
>assignvariableop_66_adam_x_deep_fm_1_embedding_31_embeddings_m:	Q
>assignvariableop_67_adam_x_deep_fm_1_embedding_32_embeddings_m:	?P
>assignvariableop_68_adam_x_deep_fm_1_embedding_33_embeddings_m:CP
>assignvariableop_69_adam_x_deep_fm_1_embedding_34_embeddings_m:Q
>assignvariableop_70_adam_x_deep_fm_1_embedding_35_embeddings_m:	?Q
>assignvariableop_71_adam_x_deep_fm_1_embedding_36_embeddings_m:	?Q
>assignvariableop_72_adam_x_deep_fm_1_embedding_37_embeddings_m:	?Q
>assignvariableop_73_adam_x_deep_fm_1_embedding_38_embeddings_m:	?P
>assignvariableop_74_adam_x_deep_fm_1_embedding_39_embeddings_m:Q
>assignvariableop_75_adam_x_deep_fm_1_embedding_40_embeddings_m:	?
Q
>assignvariableop_76_adam_x_deep_fm_1_embedding_41_embeddings_m:	?P
>assignvariableop_77_adam_x_deep_fm_1_embedding_42_embeddings_m:
Q
>assignvariableop_78_adam_x_deep_fm_1_embedding_43_embeddings_m:	?Q
>assignvariableop_79_adam_x_deep_fm_1_embedding_44_embeddings_m:	?P
>assignvariableop_80_adam_x_deep_fm_1_embedding_45_embeddings_m:Q
>assignvariableop_81_adam_x_deep_fm_1_embedding_46_embeddings_m:	?P
>assignvariableop_82_adam_x_deep_fm_1_embedding_47_embeddings_m:
P
>assignvariableop_83_adam_x_deep_fm_1_embedding_48_embeddings_m:Q
>assignvariableop_84_adam_x_deep_fm_1_embedding_49_embeddings_m:	?P
>assignvariableop_85_adam_x_deep_fm_1_embedding_50_embeddings_m:'Q
>assignvariableop_86_adam_x_deep_fm_1_embedding_51_embeddings_m:	?P
>assignvariableop_87_adam_x_deep_fm_1_linear_1_dense_5_kernel_m:'J
<assignvariableop_88_adam_x_deep_fm_1_linear_1_dense_5_bias_m:V
Cassignvariableop_89_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_m:	?@O
Aassignvariableop_90_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_m:@U
Cassignvariableop_91_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_m:@@O
Aassignvariableop_92_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_m:@U
Cassignvariableop_93_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_m:@O
Aassignvariableop_94_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_m:P
Bassignvariableop_95_adam_x_deep_fm_1_batch_normalization_1_gamma_v:'O
Aassignvariableop_96_adam_x_deep_fm_1_batch_normalization_1_beta_v:'F
/assignvariableop_97_adam_x_deep_fm_1_cin_1_w0_v:? F
/assignvariableop_98_adam_x_deep_fm_1_cin_1_w1_v:? G
5assignvariableop_99_adam_x_deep_fm_1_dense_9_kernel_v:@B
4assignvariableop_100_adam_x_deep_fm_1_dense_9_bias_v:Q
?assignvariableop_101_adam_x_deep_fm_1_embedding_26_embeddings_v:hR
?assignvariableop_102_adam_x_deep_fm_1_embedding_27_embeddings_v:	?R
?assignvariableop_103_adam_x_deep_fm_1_embedding_28_embeddings_v:	?R
?assignvariableop_104_adam_x_deep_fm_1_embedding_29_embeddings_v:	?Q
?assignvariableop_105_adam_x_deep_fm_1_embedding_30_embeddings_v:*Q
?assignvariableop_106_adam_x_deep_fm_1_embedding_31_embeddings_v:	R
?assignvariableop_107_adam_x_deep_fm_1_embedding_32_embeddings_v:	?Q
?assignvariableop_108_adam_x_deep_fm_1_embedding_33_embeddings_v:CQ
?assignvariableop_109_adam_x_deep_fm_1_embedding_34_embeddings_v:R
?assignvariableop_110_adam_x_deep_fm_1_embedding_35_embeddings_v:	?R
?assignvariableop_111_adam_x_deep_fm_1_embedding_36_embeddings_v:	?R
?assignvariableop_112_adam_x_deep_fm_1_embedding_37_embeddings_v:	?R
?assignvariableop_113_adam_x_deep_fm_1_embedding_38_embeddings_v:	?Q
?assignvariableop_114_adam_x_deep_fm_1_embedding_39_embeddings_v:R
?assignvariableop_115_adam_x_deep_fm_1_embedding_40_embeddings_v:	?
R
?assignvariableop_116_adam_x_deep_fm_1_embedding_41_embeddings_v:	?Q
?assignvariableop_117_adam_x_deep_fm_1_embedding_42_embeddings_v:
R
?assignvariableop_118_adam_x_deep_fm_1_embedding_43_embeddings_v:	?R
?assignvariableop_119_adam_x_deep_fm_1_embedding_44_embeddings_v:	?Q
?assignvariableop_120_adam_x_deep_fm_1_embedding_45_embeddings_v:R
?assignvariableop_121_adam_x_deep_fm_1_embedding_46_embeddings_v:	?Q
?assignvariableop_122_adam_x_deep_fm_1_embedding_47_embeddings_v:
Q
?assignvariableop_123_adam_x_deep_fm_1_embedding_48_embeddings_v:R
?assignvariableop_124_adam_x_deep_fm_1_embedding_49_embeddings_v:	?Q
?assignvariableop_125_adam_x_deep_fm_1_embedding_50_embeddings_v:'R
?assignvariableop_126_adam_x_deep_fm_1_embedding_51_embeddings_v:	?Q
?assignvariableop_127_adam_x_deep_fm_1_linear_1_dense_5_kernel_v:'K
=assignvariableop_128_adam_x_deep_fm_1_linear_1_dense_5_bias_v:W
Dassignvariableop_129_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_v:	?@P
Bassignvariableop_130_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_v:@V
Dassignvariableop_131_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_v:@@P
Bassignvariableop_132_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_v:@V
Dassignvariableop_133_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_v:@P
Bassignvariableop_134_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_v:
identity_136??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?>
value?>B?>?B#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB'cin_layer/w0/.ATTRIBUTES/VARIABLE_VALUEB'cin_layer/w1/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEout_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCcin_layer/w1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGout_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEout_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp8assignvariableop_x_deep_fm_1_batch_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp9assignvariableop_1_x_deep_fm_1_batch_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp@assignvariableop_2_x_deep_fm_1_batch_normalization_1_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpDassignvariableop_3_x_deep_fm_1_batch_normalization_1_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_x_deep_fm_1_cin_1_w0Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_x_deep_fm_1_cin_1_w1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_x_deep_fm_1_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_x_deep_fm_1_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_x_deep_fm_1_embedding_26_embeddingsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_x_deep_fm_1_embedding_27_embeddingsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_x_deep_fm_1_embedding_28_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp7assignvariableop_16_x_deep_fm_1_embedding_29_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp7assignvariableop_17_x_deep_fm_1_embedding_30_embeddingsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_x_deep_fm_1_embedding_31_embeddingsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp7assignvariableop_19_x_deep_fm_1_embedding_32_embeddingsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp7assignvariableop_20_x_deep_fm_1_embedding_33_embeddingsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_x_deep_fm_1_embedding_34_embeddingsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp7assignvariableop_22_x_deep_fm_1_embedding_35_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp7assignvariableop_23_x_deep_fm_1_embedding_36_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp7assignvariableop_24_x_deep_fm_1_embedding_37_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp7assignvariableop_25_x_deep_fm_1_embedding_38_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp7assignvariableop_26_x_deep_fm_1_embedding_39_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp7assignvariableop_27_x_deep_fm_1_embedding_40_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp7assignvariableop_28_x_deep_fm_1_embedding_41_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp7assignvariableop_29_x_deep_fm_1_embedding_42_embeddingsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp7assignvariableop_30_x_deep_fm_1_embedding_43_embeddingsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_x_deep_fm_1_embedding_44_embeddingsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp7assignvariableop_32_x_deep_fm_1_embedding_45_embeddingsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp7assignvariableop_33_x_deep_fm_1_embedding_46_embeddingsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_x_deep_fm_1_embedding_47_embeddingsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp7assignvariableop_35_x_deep_fm_1_embedding_48_embeddingsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_x_deep_fm_1_embedding_49_embeddingsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_x_deep_fm_1_embedding_50_embeddingsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp7assignvariableop_38_x_deep_fm_1_embedding_51_embeddingsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_x_deep_fm_1_linear_1_dense_5_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp5assignvariableop_40_x_deep_fm_1_linear_1_dense_5_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp<assignvariableop_41_x_deep_fm_1_dense_layer_1_dense_6_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp:assignvariableop_42_x_deep_fm_1_dense_layer_1_dense_6_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp<assignvariableop_43_x_deep_fm_1_dense_layer_1_dense_7_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp:assignvariableop_44_x_deep_fm_1_dense_layer_1_dense_7_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp<assignvariableop_45_x_deep_fm_1_dense_layer_1_dense_8_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp:assignvariableop_46_x_deep_fm_1_dense_layer_1_dense_8_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_true_positivesIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp"assignvariableop_50_true_negativesIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp#assignvariableop_51_false_positivesIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp#assignvariableop_52_false_negativesIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp$assignvariableop_53_true_positives_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp%assignvariableop_54_false_negatives_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOpBassignvariableop_55_adam_x_deep_fm_1_batch_normalization_1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOpAassignvariableop_56_adam_x_deep_fm_1_batch_normalization_1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp/assignvariableop_57_adam_x_deep_fm_1_cin_1_w0_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp/assignvariableop_58_adam_x_deep_fm_1_cin_1_w1_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_x_deep_fm_1_dense_9_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_x_deep_fm_1_dense_9_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp>assignvariableop_61_adam_x_deep_fm_1_embedding_26_embeddings_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_x_deep_fm_1_embedding_27_embeddings_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_x_deep_fm_1_embedding_28_embeddings_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp>assignvariableop_64_adam_x_deep_fm_1_embedding_29_embeddings_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp>assignvariableop_65_adam_x_deep_fm_1_embedding_30_embeddings_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp>assignvariableop_66_adam_x_deep_fm_1_embedding_31_embeddings_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp>assignvariableop_67_adam_x_deep_fm_1_embedding_32_embeddings_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp>assignvariableop_68_adam_x_deep_fm_1_embedding_33_embeddings_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp>assignvariableop_69_adam_x_deep_fm_1_embedding_34_embeddings_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp>assignvariableop_70_adam_x_deep_fm_1_embedding_35_embeddings_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp>assignvariableop_71_adam_x_deep_fm_1_embedding_36_embeddings_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp>assignvariableop_72_adam_x_deep_fm_1_embedding_37_embeddings_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp>assignvariableop_73_adam_x_deep_fm_1_embedding_38_embeddings_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp>assignvariableop_74_adam_x_deep_fm_1_embedding_39_embeddings_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp>assignvariableop_75_adam_x_deep_fm_1_embedding_40_embeddings_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp>assignvariableop_76_adam_x_deep_fm_1_embedding_41_embeddings_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp>assignvariableop_77_adam_x_deep_fm_1_embedding_42_embeddings_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp>assignvariableop_78_adam_x_deep_fm_1_embedding_43_embeddings_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp>assignvariableop_79_adam_x_deep_fm_1_embedding_44_embeddings_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp>assignvariableop_80_adam_x_deep_fm_1_embedding_45_embeddings_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp>assignvariableop_81_adam_x_deep_fm_1_embedding_46_embeddings_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp>assignvariableop_82_adam_x_deep_fm_1_embedding_47_embeddings_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp>assignvariableop_83_adam_x_deep_fm_1_embedding_48_embeddings_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp>assignvariableop_84_adam_x_deep_fm_1_embedding_49_embeddings_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp>assignvariableop_85_adam_x_deep_fm_1_embedding_50_embeddings_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp>assignvariableop_86_adam_x_deep_fm_1_embedding_51_embeddings_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp>assignvariableop_87_adam_x_deep_fm_1_linear_1_dense_5_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp<assignvariableop_88_adam_x_deep_fm_1_linear_1_dense_5_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOpCassignvariableop_89_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOpAassignvariableop_90_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOpCassignvariableop_91_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOpAassignvariableop_92_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOpCassignvariableop_93_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOpAassignvariableop_94_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOpBassignvariableop_95_adam_x_deep_fm_1_batch_normalization_1_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOpAassignvariableop_96_adam_x_deep_fm_1_batch_normalization_1_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp/assignvariableop_97_adam_x_deep_fm_1_cin_1_w0_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp/assignvariableop_98_adam_x_deep_fm_1_cin_1_w1_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp5assignvariableop_99_adam_x_deep_fm_1_dense_9_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp4assignvariableop_100_adam_x_deep_fm_1_dense_9_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp?assignvariableop_101_adam_x_deep_fm_1_embedding_26_embeddings_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp?assignvariableop_102_adam_x_deep_fm_1_embedding_27_embeddings_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp?assignvariableop_103_adam_x_deep_fm_1_embedding_28_embeddings_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp?assignvariableop_104_adam_x_deep_fm_1_embedding_29_embeddings_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp?assignvariableop_105_adam_x_deep_fm_1_embedding_30_embeddings_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp?assignvariableop_106_adam_x_deep_fm_1_embedding_31_embeddings_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp?assignvariableop_107_adam_x_deep_fm_1_embedding_32_embeddings_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp?assignvariableop_108_adam_x_deep_fm_1_embedding_33_embeddings_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp?assignvariableop_109_adam_x_deep_fm_1_embedding_34_embeddings_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp?assignvariableop_110_adam_x_deep_fm_1_embedding_35_embeddings_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp?assignvariableop_111_adam_x_deep_fm_1_embedding_36_embeddings_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp?assignvariableop_112_adam_x_deep_fm_1_embedding_37_embeddings_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp?assignvariableop_113_adam_x_deep_fm_1_embedding_38_embeddings_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp?assignvariableop_114_adam_x_deep_fm_1_embedding_39_embeddings_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp?assignvariableop_115_adam_x_deep_fm_1_embedding_40_embeddings_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp?assignvariableop_116_adam_x_deep_fm_1_embedding_41_embeddings_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp?assignvariableop_117_adam_x_deep_fm_1_embedding_42_embeddings_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp?assignvariableop_118_adam_x_deep_fm_1_embedding_43_embeddings_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp?assignvariableop_119_adam_x_deep_fm_1_embedding_44_embeddings_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp?assignvariableop_120_adam_x_deep_fm_1_embedding_45_embeddings_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp?assignvariableop_121_adam_x_deep_fm_1_embedding_46_embeddings_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp?assignvariableop_122_adam_x_deep_fm_1_embedding_47_embeddings_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp?assignvariableop_123_adam_x_deep_fm_1_embedding_48_embeddings_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp?assignvariableop_124_adam_x_deep_fm_1_embedding_49_embeddings_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp?assignvariableop_125_adam_x_deep_fm_1_embedding_50_embeddings_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp?assignvariableop_126_adam_x_deep_fm_1_embedding_51_embeddings_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp?assignvariableop_127_adam_x_deep_fm_1_linear_1_dense_5_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp=assignvariableop_128_adam_x_deep_fm_1_linear_1_dense_5_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOpDassignvariableop_129_adam_x_deep_fm_1_dense_layer_1_dense_6_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOpBassignvariableop_130_adam_x_deep_fm_1_dense_layer_1_dense_6_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOpDassignvariableop_131_adam_x_deep_fm_1_dense_layer_1_dense_7_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOpBassignvariableop_132_adam_x_deep_fm_1_dense_layer_1_dense_7_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOpDassignvariableop_133_adam_x_deep_fm_1_dense_layer_1_dense_8_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOpBassignvariableop_134_adam_x_deep_fm_1_dense_layer_1_dense_8_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_135Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_135i
Identity_136IdentityIdentity_135:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_136?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_136Identity_136:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
G__inference_embedding_42_layer_call_and_return_conditional_losses_85663

inputs(
embedding_lookup_85657:

identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85657Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85657*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85657*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?#
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88791

inputsK
=batch_normalization_1_assignmovingavg_readvariableop_resource:'M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:'@
2batch_normalization_1_cast_readvariableop_resource:'B
4batch_normalization_1_cast_1_readvariableop_resource:'A
/linear_1_dense_5_matmul_readvariableop_resource:'>
0linear_1_dense_5_biasadd_readvariableop_resource:5
#embedding_26_embedding_lookup_88404:h6
#embedding_27_embedding_lookup_88414:	?6
#embedding_28_embedding_lookup_88424:	?6
#embedding_29_embedding_lookup_88434:	?5
#embedding_30_embedding_lookup_88444:*5
#embedding_31_embedding_lookup_88454:	6
#embedding_32_embedding_lookup_88464:	?5
#embedding_33_embedding_lookup_88474:C5
#embedding_34_embedding_lookup_88484:6
#embedding_35_embedding_lookup_88494:	?6
#embedding_36_embedding_lookup_88504:	?6
#embedding_37_embedding_lookup_88514:	?6
#embedding_38_embedding_lookup_88524:	?5
#embedding_39_embedding_lookup_88534:6
#embedding_40_embedding_lookup_88544:	?
6
#embedding_41_embedding_lookup_88554:	?5
#embedding_42_embedding_lookup_88564:
6
#embedding_43_embedding_lookup_88574:	?6
#embedding_44_embedding_lookup_88584:	?5
#embedding_45_embedding_lookup_88594:6
#embedding_46_embedding_lookup_88604:	?5
#embedding_47_embedding_lookup_88614:
5
#embedding_48_embedding_lookup_88624:6
#embedding_49_embedding_lookup_88634:	?5
#embedding_50_embedding_lookup_88644:'6
#embedding_51_embedding_lookup_88654:	?H
1cin_1_conv1d_expanddims_1_readvariableop_resource:? J
3cin_1_conv1d_1_expanddims_1_readvariableop_resource:? G
4dense_layer_1_dense_6_matmul_readvariableop_resource:	?@C
5dense_layer_1_dense_6_biasadd_readvariableop_resource:@F
4dense_layer_1_dense_7_matmul_readvariableop_resource:@@C
5dense_layer_1_dense_7_biasadd_readvariableop_resource:@F
4dense_layer_1_dense_8_matmul_readvariableop_resource:@C
5dense_layer_1_dense_8_biasadd_readvariableop_resource:8
&dense_9_matmul_readvariableop_resource:@5
'dense_9_biasadd_readvariableop_resource:
identity??%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?)batch_normalization_1/Cast/ReadVariableOp?+batch_normalization_1/Cast_1/ReadVariableOp?(cin_1/conv1d/ExpandDims_1/ReadVariableOp?*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp?dense_9/MatMul/ReadVariableOp?,dense_layer_1/dense_6/BiasAdd/ReadVariableOp?+dense_layer_1/dense_6/MatMul/ReadVariableOp?,dense_layer_1/dense_7/BiasAdd/ReadVariableOp?+dense_layer_1/dense_7/MatMul/ReadVariableOp?,dense_layer_1/dense_8/BiasAdd/ReadVariableOp?+dense_layer_1/dense_8/MatMul/ReadVariableOp?embedding_26/embedding_lookup?embedding_27/embedding_lookup?embedding_28/embedding_lookup?embedding_29/embedding_lookup?embedding_30/embedding_lookup?embedding_31/embedding_lookup?embedding_32/embedding_lookup?embedding_33/embedding_lookup?embedding_34/embedding_lookup?embedding_35/embedding_lookup?embedding_36/embedding_lookup?embedding_37/embedding_lookup?embedding_38/embedding_lookup?embedding_39/embedding_lookup?embedding_40/embedding_lookup?embedding_41/embedding_lookup?embedding_42/embedding_lookup?embedding_43/embedding_lookup?embedding_44/embedding_lookup?embedding_45/embedding_lookup?embedding_46/embedding_lookup?embedding_47/embedding_lookup?embedding_48/embedding_lookup?embedding_49/embedding_lookup?embedding_50/embedding_lookup?embedding_51/embedding_lookup?'linear_1/dense_5/BiasAdd/ReadVariableOp?&linear_1/dense_5/MatMul/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMeaninputs=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes

:'2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceinputs3batch_normalization_1/moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????'21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:'*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:'2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:'2+
)batch_normalization_1/AssignMovingAvg/mul?
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg?
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:'*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:'2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:'2-
+batch_normalization_1/AssignMovingAvg_1/mul?
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1?
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes
:'*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp?
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
:'*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:'2'
%batch_normalization_1/batchnorm/Rsqrt?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mulinputs'batch_normalization_1/batchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:'2'
%batch_normalization_1/batchnorm/mul_2?
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2'
%batch_normalization_1/batchnorm/add_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSlice)batch_normalization_1/batchnorm/add_1:z:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlice)batch_normalization_1/batchnorm/add_1:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice_1?
&linear_1/dense_5/MatMul/ReadVariableOpReadVariableOp/linear_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:'*
dtype02(
&linear_1/dense_5/MatMul/ReadVariableOp?
linear_1/dense_5/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0.linear_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
linear_1/dense_5/MatMul?
'linear_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp0linear_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'linear_1/dense_5/BiasAdd/ReadVariableOp?
linear_1/dense_5/BiasAddBiasAdd!linear_1/dense_5/MatMul:product:0/linear_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
linear_1/dense_5/BiasAdd
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
embedding_26/CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_26/Cast?
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_88404embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/88404*'
_output_shapes
:?????????*
dtype02
embedding_26/embedding_lookup?
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/88404*'
_output_shapes
:?????????2(
&embedding_26/embedding_lookup/Identity?
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_26/embedding_lookup/Identity_1
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
embedding_27/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_27/Cast?
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_88414embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/88414*'
_output_shapes
:?????????*
dtype02
embedding_27/embedding_lookup?
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/88414*'
_output_shapes
:?????????2(
&embedding_27/embedding_lookup/Identity?
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_27/embedding_lookup/Identity_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4?
embedding_28/CastCaststrided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_28/Cast?
embedding_28/embedding_lookupResourceGather#embedding_28_embedding_lookup_88424embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_28/embedding_lookup/88424*'
_output_shapes
:?????????*
dtype02
embedding_28/embedding_lookup?
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_28/embedding_lookup/88424*'
_output_shapes
:?????????2(
&embedding_28/embedding_lookup/Identity?
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_28/embedding_lookup/Identity_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5?
embedding_29/CastCaststrided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_29/Cast?
embedding_29/embedding_lookupResourceGather#embedding_29_embedding_lookup_88434embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_29/embedding_lookup/88434*'
_output_shapes
:?????????*
dtype02
embedding_29/embedding_lookup?
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_29/embedding_lookup/88434*'
_output_shapes
:?????????2(
&embedding_29/embedding_lookup/Identity?
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_29/embedding_lookup/Identity_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6?
embedding_30/CastCaststrided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_30/Cast?
embedding_30/embedding_lookupResourceGather#embedding_30_embedding_lookup_88444embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_30/embedding_lookup/88444*'
_output_shapes
:?????????*
dtype02
embedding_30/embedding_lookup?
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_30/embedding_lookup/88444*'
_output_shapes
:?????????2(
&embedding_30/embedding_lookup/Identity?
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_30/embedding_lookup/Identity_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7?
embedding_31/CastCaststrided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_31/Cast?
embedding_31/embedding_lookupResourceGather#embedding_31_embedding_lookup_88454embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_31/embedding_lookup/88454*'
_output_shapes
:?????????*
dtype02
embedding_31/embedding_lookup?
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_31/embedding_lookup/88454*'
_output_shapes
:?????????2(
&embedding_31/embedding_lookup/Identity?
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_31/embedding_lookup/Identity_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8?
embedding_32/CastCaststrided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_32/Cast?
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_88464embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/88464*'
_output_shapes
:?????????*
dtype02
embedding_32/embedding_lookup?
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/88464*'
_output_shapes
:?????????2(
&embedding_32/embedding_lookup/Identity?
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_32/embedding_lookup/Identity_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9?
embedding_33/CastCaststrided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_33/Cast?
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_88474embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/88474*'
_output_shapes
:?????????*
dtype02
embedding_33/embedding_lookup?
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/88474*'
_output_shapes
:?????????2(
&embedding_33/embedding_lookup/Identity?
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_33/embedding_lookup/Identity_1?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10?
embedding_34/CastCaststrided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_34/Cast?
embedding_34/embedding_lookupResourceGather#embedding_34_embedding_lookup_88484embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_34/embedding_lookup/88484*'
_output_shapes
:?????????*
dtype02
embedding_34/embedding_lookup?
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_34/embedding_lookup/88484*'
_output_shapes
:?????????2(
&embedding_34/embedding_lookup/Identity?
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_34/embedding_lookup/Identity_1?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11?
embedding_35/CastCaststrided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_35/Cast?
embedding_35/embedding_lookupResourceGather#embedding_35_embedding_lookup_88494embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_35/embedding_lookup/88494*'
_output_shapes
:?????????*
dtype02
embedding_35/embedding_lookup?
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_35/embedding_lookup/88494*'
_output_shapes
:?????????2(
&embedding_35/embedding_lookup/Identity?
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_35/embedding_lookup/Identity_1?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stack?
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1?
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2?
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12?
embedding_36/CastCaststrided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_36/Cast?
embedding_36/embedding_lookupResourceGather#embedding_36_embedding_lookup_88504embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_36/embedding_lookup/88504*'
_output_shapes
:?????????*
dtype02
embedding_36/embedding_lookup?
&embedding_36/embedding_lookup/IdentityIdentity&embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_36/embedding_lookup/88504*'
_output_shapes
:?????????2(
&embedding_36/embedding_lookup/Identity?
(embedding_36/embedding_lookup/Identity_1Identity/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_36/embedding_lookup/Identity_1?
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack?
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1?
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2?
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13?
embedding_37/CastCaststrided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_37/Cast?
embedding_37/embedding_lookupResourceGather#embedding_37_embedding_lookup_88514embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_37/embedding_lookup/88514*'
_output_shapes
:?????????*
dtype02
embedding_37/embedding_lookup?
&embedding_37/embedding_lookup/IdentityIdentity&embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_37/embedding_lookup/88514*'
_output_shapes
:?????????2(
&embedding_37/embedding_lookup/Identity?
(embedding_37/embedding_lookup/Identity_1Identity/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_37/embedding_lookup/Identity_1?
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack?
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1?
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2?
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14?
embedding_38/CastCaststrided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_38/Cast?
embedding_38/embedding_lookupResourceGather#embedding_38_embedding_lookup_88524embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_38/embedding_lookup/88524*'
_output_shapes
:?????????*
dtype02
embedding_38/embedding_lookup?
&embedding_38/embedding_lookup/IdentityIdentity&embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_38/embedding_lookup/88524*'
_output_shapes
:?????????2(
&embedding_38/embedding_lookup/Identity?
(embedding_38/embedding_lookup/Identity_1Identity/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_38/embedding_lookup/Identity_1?
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack?
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1?
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2?
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15?
embedding_39/CastCaststrided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_39/Cast?
embedding_39/embedding_lookupResourceGather#embedding_39_embedding_lookup_88534embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_39/embedding_lookup/88534*'
_output_shapes
:?????????*
dtype02
embedding_39/embedding_lookup?
&embedding_39/embedding_lookup/IdentityIdentity&embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_39/embedding_lookup/88534*'
_output_shapes
:?????????2(
&embedding_39/embedding_lookup/Identity?
(embedding_39/embedding_lookup/Identity_1Identity/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_39/embedding_lookup/Identity_1?
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack?
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1?
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2?
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16?
embedding_40/CastCaststrided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_40/Cast?
embedding_40/embedding_lookupResourceGather#embedding_40_embedding_lookup_88544embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_40/embedding_lookup/88544*'
_output_shapes
:?????????*
dtype02
embedding_40/embedding_lookup?
&embedding_40/embedding_lookup/IdentityIdentity&embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_40/embedding_lookup/88544*'
_output_shapes
:?????????2(
&embedding_40/embedding_lookup/Identity?
(embedding_40/embedding_lookup/Identity_1Identity/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_40/embedding_lookup/Identity_1?
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack?
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1?
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2?
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17?
embedding_41/CastCaststrided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_41/Cast?
embedding_41/embedding_lookupResourceGather#embedding_41_embedding_lookup_88554embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_41/embedding_lookup/88554*'
_output_shapes
:?????????*
dtype02
embedding_41/embedding_lookup?
&embedding_41/embedding_lookup/IdentityIdentity&embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_41/embedding_lookup/88554*'
_output_shapes
:?????????2(
&embedding_41/embedding_lookup/Identity?
(embedding_41/embedding_lookup/Identity_1Identity/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_41/embedding_lookup/Identity_1?
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack?
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1?
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2?
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18?
embedding_42/CastCaststrided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_42/Cast?
embedding_42/embedding_lookupResourceGather#embedding_42_embedding_lookup_88564embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_42/embedding_lookup/88564*'
_output_shapes
:?????????*
dtype02
embedding_42/embedding_lookup?
&embedding_42/embedding_lookup/IdentityIdentity&embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_42/embedding_lookup/88564*'
_output_shapes
:?????????2(
&embedding_42/embedding_lookup/Identity?
(embedding_42/embedding_lookup/Identity_1Identity/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_42/embedding_lookup/Identity_1?
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack?
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1?
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2?
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19?
embedding_43/CastCaststrided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_43/Cast?
embedding_43/embedding_lookupResourceGather#embedding_43_embedding_lookup_88574embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_43/embedding_lookup/88574*'
_output_shapes
:?????????*
dtype02
embedding_43/embedding_lookup?
&embedding_43/embedding_lookup/IdentityIdentity&embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_43/embedding_lookup/88574*'
_output_shapes
:?????????2(
&embedding_43/embedding_lookup/Identity?
(embedding_43/embedding_lookup/Identity_1Identity/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_43/embedding_lookup/Identity_1?
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack?
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1?
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2?
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20?
embedding_44/CastCaststrided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_44/Cast?
embedding_44/embedding_lookupResourceGather#embedding_44_embedding_lookup_88584embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_44/embedding_lookup/88584*'
_output_shapes
:?????????*
dtype02
embedding_44/embedding_lookup?
&embedding_44/embedding_lookup/IdentityIdentity&embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_44/embedding_lookup/88584*'
_output_shapes
:?????????2(
&embedding_44/embedding_lookup/Identity?
(embedding_44/embedding_lookup/Identity_1Identity/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_44/embedding_lookup/Identity_1?
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack?
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1?
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2?
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21?
embedding_45/CastCaststrided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_45/Cast?
embedding_45/embedding_lookupResourceGather#embedding_45_embedding_lookup_88594embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_45/embedding_lookup/88594*'
_output_shapes
:?????????*
dtype02
embedding_45/embedding_lookup?
&embedding_45/embedding_lookup/IdentityIdentity&embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_45/embedding_lookup/88594*'
_output_shapes
:?????????2(
&embedding_45/embedding_lookup/Identity?
(embedding_45/embedding_lookup/Identity_1Identity/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_45/embedding_lookup/Identity_1?
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack?
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1?
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2?
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22?
embedding_46/CastCaststrided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_46/Cast?
embedding_46/embedding_lookupResourceGather#embedding_46_embedding_lookup_88604embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_46/embedding_lookup/88604*'
_output_shapes
:?????????*
dtype02
embedding_46/embedding_lookup?
&embedding_46/embedding_lookup/IdentityIdentity&embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_46/embedding_lookup/88604*'
_output_shapes
:?????????2(
&embedding_46/embedding_lookup/Identity?
(embedding_46/embedding_lookup/Identity_1Identity/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_46/embedding_lookup/Identity_1?
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack?
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1?
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2?
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23?
embedding_47/CastCaststrided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_47/Cast?
embedding_47/embedding_lookupResourceGather#embedding_47_embedding_lookup_88614embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_47/embedding_lookup/88614*'
_output_shapes
:?????????*
dtype02
embedding_47/embedding_lookup?
&embedding_47/embedding_lookup/IdentityIdentity&embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_47/embedding_lookup/88614*'
_output_shapes
:?????????2(
&embedding_47/embedding_lookup/Identity?
(embedding_47/embedding_lookup/Identity_1Identity/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_47/embedding_lookup/Identity_1?
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack?
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1?
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2?
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24?
embedding_48/CastCaststrided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_48/Cast?
embedding_48/embedding_lookupResourceGather#embedding_48_embedding_lookup_88624embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_48/embedding_lookup/88624*'
_output_shapes
:?????????*
dtype02
embedding_48/embedding_lookup?
&embedding_48/embedding_lookup/IdentityIdentity&embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_48/embedding_lookup/88624*'
_output_shapes
:?????????2(
&embedding_48/embedding_lookup/Identity?
(embedding_48/embedding_lookup/Identity_1Identity/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_48/embedding_lookup/Identity_1?
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack?
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1?
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2?
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25?
embedding_49/CastCaststrided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_49/Cast?
embedding_49/embedding_lookupResourceGather#embedding_49_embedding_lookup_88634embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_49/embedding_lookup/88634*'
_output_shapes
:?????????*
dtype02
embedding_49/embedding_lookup?
&embedding_49/embedding_lookup/IdentityIdentity&embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_49/embedding_lookup/88634*'
_output_shapes
:?????????2(
&embedding_49/embedding_lookup/Identity?
(embedding_49/embedding_lookup/Identity_1Identity/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_49/embedding_lookup/Identity_1?
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack?
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1?
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2?
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26?
embedding_50/CastCaststrided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_50/Cast?
embedding_50/embedding_lookupResourceGather#embedding_50_embedding_lookup_88644embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_50/embedding_lookup/88644*'
_output_shapes
:?????????*
dtype02
embedding_50/embedding_lookup?
&embedding_50/embedding_lookup/IdentityIdentity&embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_50/embedding_lookup/88644*'
_output_shapes
:?????????2(
&embedding_50/embedding_lookup/Identity?
(embedding_50/embedding_lookup/Identity_1Identity/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_50/embedding_lookup/Identity_1?
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack?
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1?
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2?
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27?
embedding_51/CastCaststrided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:?????????2
embedding_51/Cast?
embedding_51/embedding_lookupResourceGather#embedding_51_embedding_lookup_88654embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_51/embedding_lookup/88654*'
_output_shapes
:?????????*
dtype02
embedding_51/embedding_lookup?
&embedding_51/embedding_lookup/IdentityIdentity&embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_51/embedding_lookup/88654*'
_output_shapes
:?????????2(
&embedding_51/embedding_lookup/Identity?
(embedding_51/embedding_lookup/Identity_1Identity/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2*
(embedding_51/embedding_lookup/Identity_1?
packedPack1embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:01embedding_28/embedding_lookup/Identity_1:output:01embedding_29/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:01embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:01embedding_34/embedding_lookup/Identity_1:output:01embedding_35/embedding_lookup/Identity_1:output:01embedding_36/embedding_lookup/Identity_1:output:01embedding_37/embedding_lookup/Identity_1:output:01embedding_38/embedding_lookup/Identity_1:output:01embedding_39/embedding_lookup/Identity_1:output:01embedding_40/embedding_lookup/Identity_1:output:01embedding_41/embedding_lookup/Identity_1:output:01embedding_42/embedding_lookup/Identity_1:output:01embedding_43/embedding_lookup/Identity_1:output:01embedding_44/embedding_lookup/Identity_1:output:01embedding_45/embedding_lookup/Identity_1:output:01embedding_46/embedding_lookup/Identity_1:output:01embedding_47/embedding_lookup/Identity_1:output:01embedding_48/embedding_lookup/Identity_1:output:01embedding_49/embedding_lookup/Identity_1:output:01embedding_50/embedding_lookup/Identity_1:output:01embedding_51/embedding_lookup/Identity_1:output:0*
N*
T0*+
_output_shapes
:?????????2
packedu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposepacked:output:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposey
cin_1/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split/split_dim?
cin_1/splitSplitcin_1/split/split_dim:output:0transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
cin_1/split}
cin_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split_1/split_dim?
cin_1/split_1Split cin_1/split_1/split_dim:output:0transpose:y:0*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
cin_1/split_1?
cin_1/MatMul/aPackcin_1/split:output:0cin_1/split:output:1cin_1/split:output:2cin_1/split:output:3cin_1/split:output:4cin_1/split:output:5cin_1/split:output:6cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul/a?
cin_1/MatMul/bPackcin_1/split_1:output:0cin_1/split_1:output:1cin_1/split_1:output:2cin_1/split_1:output:3cin_1/split_1:output:4cin_1/split_1:output:5cin_1/split_1:output:6cin_1/split_1:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul/b?
cin_1/MatMulBatchMatMulV2cin_1/MatMul/a:output:0cin_1/MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
cin_1/MatMul
cin_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
cin_1/Reshape/shape?
cin_1/ReshapeReshapecin_1/MatMul:output:0cin_1/Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2
cin_1/Reshape?
cin_1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose/perm?
cin_1/transpose	Transposecin_1/Reshape:output:0cin_1/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
cin_1/transpose?
cin_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/conv1d/ExpandDims/dim?
cin_1/conv1d/ExpandDims
ExpandDimscin_1/transpose:y:0$cin_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
cin_1/conv1d/ExpandDims?
(cin_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02*
(cin_1/conv1d/ExpandDims_1/ReadVariableOp?
cin_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
cin_1/conv1d/ExpandDims_1/dim?
cin_1/conv1d/ExpandDims_1
ExpandDims0cin_1/conv1d/ExpandDims_1/ReadVariableOp:value:0&cin_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
cin_1/conv1d/ExpandDims_1?
cin_1/conv1dConv2D cin_1/conv1d/ExpandDims:output:0"cin_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
cin_1/conv1d?
cin_1/conv1d/SqueezeSqueezecin_1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
cin_1/conv1d/Squeeze?
cin_1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_1/perm?
cin_1/transpose_1	Transposecin_1/conv1d/Squeeze:output:0cin_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
cin_1/transpose_1}
cin_1/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/split_2/split_dim?
cin_1/split_2Split cin_1/split_2/split_dim:output:0cin_1/transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2
cin_1/split_2?
cin_1/MatMul_1/aPackcin_1/split:output:0cin_1/split:output:1cin_1/split:output:2cin_1/split:output:3cin_1/split:output:4cin_1/split:output:5cin_1/split:output:6cin_1/split:output:7*
N*
T0*/
_output_shapes
:?????????2
cin_1/MatMul_1/a?
cin_1/MatMul_1/bPackcin_1/split_2:output:0cin_1/split_2:output:1cin_1/split_2:output:2cin_1/split_2:output:3cin_1/split_2:output:4cin_1/split_2:output:5cin_1/split_2:output:6cin_1/split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2
cin_1/MatMul_1/b?
cin_1/MatMul_1BatchMatMulV2cin_1/MatMul_1/a:output:0cin_1/MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2
cin_1/MatMul_1?
cin_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
cin_1/Reshape_1/shape?
cin_1/Reshape_1Reshapecin_1/MatMul_1:output:0cin_1/Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
cin_1/Reshape_1?
cin_1/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_2/perm?
cin_1/transpose_2	Transposecin_1/Reshape_1:output:0cin_1/transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
cin_1/transpose_2?
cin_1/conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/conv1d_1/ExpandDims/dim?
cin_1/conv1d_1/ExpandDims
ExpandDimscin_1/transpose_2:y:0&cin_1/conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
cin_1/conv1d_1/ExpandDims?
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02,
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp?
cin_1/conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2!
cin_1/conv1d_1/ExpandDims_1/dim?
cin_1/conv1d_1/ExpandDims_1
ExpandDims2cin_1/conv1d_1/ExpandDims_1/ReadVariableOp:value:0(cin_1/conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
cin_1/conv1d_1/ExpandDims_1?
cin_1/conv1d_1Conv2D"cin_1/conv1d_1/ExpandDims:output:0$cin_1/conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
cin_1/conv1d_1?
cin_1/conv1d_1/SqueezeSqueezecin_1/conv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
cin_1/conv1d_1/Squeeze?
cin_1/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cin_1/transpose_3/perm?
cin_1/transpose_3	Transposecin_1/conv1d_1/Squeeze:output:0cin_1/transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
cin_1/transpose_3h
cin_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
cin_1/concat/axis?
cin_1/concatConcatV2cin_1/transpose_1:y:0cin_1/transpose_3:y:0cin_1/concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
cin_1/concat?
cin_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cin_1/Sum/reduction_indices?
	cin_1/SumSumcin_1/concat:output:0$cin_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
	cin_1/Sumo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Reshape/shapew
ReshapeReshapetranspose:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:??????????2	
Reshape\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice:output:0Reshape:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:??????????2
concat?
+dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02-
+dense_layer_1/dense_6/MatMul/ReadVariableOp?
dense_layer_1/dense_6/MatMulMatMulconcat:output:03dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/MatMul?
,dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp?
dense_layer_1/dense_6/BiasAddBiasAdd&dense_layer_1/dense_6/MatMul:product:04dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/BiasAdd?
dense_layer_1/dense_6/ReluRelu&dense_layer_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_6/Relu?
+dense_layer_1/dense_7/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02-
+dense_layer_1/dense_7/MatMul/ReadVariableOp?
dense_layer_1/dense_7/MatMulMatMul(dense_layer_1/dense_6/Relu:activations:03dense_layer_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/MatMul?
,dense_layer_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,dense_layer_1/dense_7/BiasAdd/ReadVariableOp?
dense_layer_1/dense_7/BiasAddBiasAdd&dense_layer_1/dense_7/MatMul:product:04dense_layer_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/BiasAdd?
dense_layer_1/dense_7/ReluRelu&dense_layer_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_layer_1/dense_7/Relu?
+dense_layer_1/dense_8/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02-
+dense_layer_1/dense_8/MatMul/ReadVariableOp?
dense_layer_1/dense_8/MatMulMatMul(dense_layer_1/dense_7/Relu:activations:03dense_layer_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_1/dense_8/MatMul?
,dense_layer_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_layer_1/dense_8/BiasAdd/ReadVariableOp?
dense_layer_1/dense_8/BiasAddBiasAdd&dense_layer_1/dense_8/MatMul:product:04dense_layer_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_layer_1/dense_8/BiasAdd|
addAddV2!linear_1/dense_5/BiasAdd:output:0cin_1/Sum:output:0*
T0*'
_output_shapes
:?????????@2
addz
add_1AddV2add:z:0&dense_layer_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
add_1?
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_9/MatMul/ReadVariableOp?
dense_9/MatMulMatMul	add_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/MatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_9/BiasAddi
SigmoidSigmoiddense_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp1cin_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp3cin_1_conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1f
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp)^cin_1/conv1d/ExpandDims_1/ReadVariableOp+^cin_1/conv1d_1/ExpandDims_1/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp-^dense_layer_1/dense_6/BiasAdd/ReadVariableOp,^dense_layer_1/dense_6/MatMul/ReadVariableOp-^dense_layer_1/dense_7/BiasAdd/ReadVariableOp,^dense_layer_1/dense_7/MatMul/ReadVariableOp-^dense_layer_1/dense_8/BiasAdd/ReadVariableOp,^dense_layer_1/dense_8/MatMul/ReadVariableOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup^embedding_36/embedding_lookup^embedding_37/embedding_lookup^embedding_38/embedding_lookup^embedding_39/embedding_lookup^embedding_40/embedding_lookup^embedding_41/embedding_lookup^embedding_42/embedding_lookup^embedding_43/embedding_lookup^embedding_44/embedding_lookup^embedding_45/embedding_lookup^embedding_46/embedding_lookup^embedding_47/embedding_lookup^embedding_48/embedding_lookup^embedding_49/embedding_lookup^embedding_50/embedding_lookup^embedding_51/embedding_lookup(^linear_1/dense_5/BiasAdd/ReadVariableOp'^linear_1/dense_5/MatMul/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2T
(cin_1/conv1d/ExpandDims_1/ReadVariableOp(cin_1/conv1d/ExpandDims_1/ReadVariableOp2X
*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp*cin_1/conv1d_1/ExpandDims_1/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2\
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp,dense_layer_1/dense_6/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_6/MatMul/ReadVariableOp+dense_layer_1/dense_6/MatMul/ReadVariableOp2\
,dense_layer_1/dense_7/BiasAdd/ReadVariableOp,dense_layer_1/dense_7/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_7/MatMul/ReadVariableOp+dense_layer_1/dense_7/MatMul/ReadVariableOp2\
,dense_layer_1/dense_8/BiasAdd/ReadVariableOp,dense_layer_1/dense_8/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_8/MatMul/ReadVariableOp+dense_layer_1/dense_8/MatMul/ReadVariableOp2>
embedding_26/embedding_lookupembedding_26/embedding_lookup2>
embedding_27/embedding_lookupembedding_27/embedding_lookup2>
embedding_28/embedding_lookupembedding_28/embedding_lookup2>
embedding_29/embedding_lookupembedding_29/embedding_lookup2>
embedding_30/embedding_lookupembedding_30/embedding_lookup2>
embedding_31/embedding_lookupembedding_31/embedding_lookup2>
embedding_32/embedding_lookupembedding_32/embedding_lookup2>
embedding_33/embedding_lookupembedding_33/embedding_lookup2>
embedding_34/embedding_lookupembedding_34/embedding_lookup2>
embedding_35/embedding_lookupembedding_35/embedding_lookup2>
embedding_36/embedding_lookupembedding_36/embedding_lookup2>
embedding_37/embedding_lookupembedding_37/embedding_lookup2>
embedding_38/embedding_lookupembedding_38/embedding_lookup2>
embedding_39/embedding_lookupembedding_39/embedding_lookup2>
embedding_40/embedding_lookupembedding_40/embedding_lookup2>
embedding_41/embedding_lookupembedding_41/embedding_lookup2>
embedding_42/embedding_lookupembedding_42/embedding_lookup2>
embedding_43/embedding_lookupembedding_43/embedding_lookup2>
embedding_44/embedding_lookupembedding_44/embedding_lookup2>
embedding_45/embedding_lookupembedding_45/embedding_lookup2>
embedding_46/embedding_lookupembedding_46/embedding_lookup2>
embedding_47/embedding_lookupembedding_47/embedding_lookup2>
embedding_48/embedding_lookupembedding_48/embedding_lookup2>
embedding_49/embedding_lookupembedding_49/embedding_lookup2>
embedding_50/embedding_lookupembedding_50/embedding_lookup2>
embedding_51/embedding_lookupembedding_51/embedding_lookup2R
'linear_1/dense_5/BiasAdd/ReadVariableOp'linear_1/dense_5/BiasAdd/ReadVariableOp2P
&linear_1/dense_5/MatMul/ReadVariableOp&linear_1/dense_5/MatMul/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
,__inference_embedding_28_layer_call_fn_89338

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_854112
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

+__inference_x_deep_fm_1_layer_call_fn_87837

inputs
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
	unknown_3:'
	unknown_4:
	unknown_5:h
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:*

unknown_10:	

unknown_11:	?

unknown_12:C

unknown_13:

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?


unknown_20:	?

unknown_21:


unknown_22:	?

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:


unknown_27:

unknown_28:	?

unknown_29:'

unknown_30:	?!

unknown_31:? !

unknown_32:? 

unknown_33:	?@

unknown_34:@

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:

unknown_39:@

unknown_40:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_860302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?
?
,__inference_embedding_46_layer_call_fn_89644

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_857352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?

+__inference_x_deep_fm_1_layer_call_fn_86117
input_1
unknown:'
	unknown_0:'
	unknown_1:'
	unknown_2:'
	unknown_3:'
	unknown_4:
	unknown_5:h
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:*

unknown_10:	

unknown_11:	?

unknown_12:C

unknown_13:

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:

unknown_19:	?


unknown_20:	?

unknown_21:


unknown_22:	?

unknown_23:	?

unknown_24:

unknown_25:	?

unknown_26:


unknown_27:

unknown_28:	?

unknown_29:'

unknown_30:	?!

unknown_31:? !

unknown_32:? 

unknown_33:	?@

unknown_34:@

unknown_35:@@

unknown_36:@

unknown_37:@

unknown_38:

unknown_39:@

unknown_40:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_860302
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:?????????': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????'
!
_user_specified_name	input_1
?
?
,__inference_embedding_37_layer_call_fn_89491

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_855732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
?
@__inference_cin_1_layer_call_and_return_conditional_losses_85931

inputsB
+conv1d_expanddims_1_readvariableop_resource:? D
-conv1d_1_expanddims_1_readvariableop_resource:? 
identity??"conv1d/ExpandDims_1/ReadVariableOp?$conv1d_1/ExpandDims_1/ReadVariableOp?3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split/split_dim?
splitSplitsplit/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2
splitq
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_1/split_dim?
split_1Splitsplit_1/split_dim:output:0inputs*
T0*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*
	num_split2	
split_1?
MatMul/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/a?
MatMul/bPacksplit_1:output:0split_1:output:1split_1:output:2split_1:output:3split_1:output:4split_1:output:5split_1:output:6split_1:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul/b?
MatMulBatchMatMulV2MatMul/a:output:0MatMul/b:output:0*
T0*/
_output_shapes
:?????????*
adj_y(2
MatMuls
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ?????  2
Reshape/shape}
ReshapeReshapeMatMul:output:0Reshape/shape:output:0*
T0*,
_output_shapes
:??????????2	
Reshapeu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
	transposey
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimstranspose:y:0conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeezey
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transposeconv1d/Squeeze:output:0transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_1q
split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
split_2/split_dim?
split_2Splitsplit_2/split_dim:output:0transpose_1:y:0*
T0*?
_output_shapes?
?:????????? :????????? :????????? :????????? :????????? :????????? :????????? :????????? *
	num_split2	
split_2?

MatMul_1/aPacksplit:output:0split:output:1split:output:2split:output:3split:output:4split:output:5split:output:6split:output:7*
N*
T0*/
_output_shapes
:?????????2

MatMul_1/a?

MatMul_1/bPacksplit_2:output:0split_2:output:1split_2:output:2split_2:output:3split_2:output:4split_2:output:5split_2:output:6split_2:output:7*
N*
T0*/
_output_shapes
:????????? 2

MatMul_1/b?
MatMul_1BatchMatMulV2MatMul_1/a:output:0MatMul_1/b:output:0*
T0*/
_output_shapes
:????????? *
adj_y(2

MatMul_1w
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   ????@  2
Reshape_1/shape?
	Reshape_1ReshapeMatMul_1:output:0Reshape_1/shape:output:0*
T0*,
_output_shapes
:??????????2
	Reshape_1y
transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_2/perm?
transpose_2	TransposeReshape_1:output:0transpose_2/perm:output:0*
T0*,
_output_shapes
:??????????2
transpose_2}
conv1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d_1/ExpandDims/dim?
conv1d_1/ExpandDims
ExpandDimstranspose_2:y:0 conv1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d_1/ExpandDims?
$conv1d_1/ExpandDims_1/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype02&
$conv1d_1/ExpandDims_1/ReadVariableOpx
conv1d_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_1/ExpandDims_1/dim?
conv1d_1/ExpandDims_1
ExpandDims,conv1d_1/ExpandDims_1/ReadVariableOp:value:0"conv1d_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:? 2
conv1d_1/ExpandDims_1?
conv1d_1Conv2Dconv1d_1/ExpandDims:output:0conv1d_1/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2

conv1d_1?
conv1d_1/SqueezeSqueezeconv1d_1:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d_1/Squeezey
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_3/perm?
transpose_3	Transposeconv1d_1/Squeeze:output:0transpose_3/perm:output:0*
T0*+
_output_shapes
:????????? 2
transpose_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2transpose_1:y:0transpose_3:y:0concat/axis:output:0*
N*
T0*+
_output_shapes
:?????????@2
concaty
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Sum/reduction_indicest
SumSumconcat:output:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1?
&x_deep_fm_1/cin_1/w1/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w1/Regularizer/Const?
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w1/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Abs?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w1/Regularizer/SumSum(x_deep_fm_1/cin_1/w1/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/Sum?
&x_deep_fm_1/cin_1/w1/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w1/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w1/Regularizer/mulMul/x_deep_fm_1/cin_1/w1/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w1/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/mul?
$x_deep_fm_1/cin_1/w1/Regularizer/addAddV2/x_deep_fm_1/cin_1/w1/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w1/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w1/Regularizer/add?
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOpReadVariableOp-conv1d_1_expanddims_1_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w1/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w1/Regularizer/Square?
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w1/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w1/Regularizer/Square:y:01x_deep_fm_1/cin_1/w1/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w1/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w1/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w1/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w1/Regularizer/add:z:0*x_deep_fm_1/cin_1/w1/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w1/Regularizer/add_1g
IdentityIdentitySum:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity?
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp%^conv1d_1/ExpandDims_1/ReadVariableOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp4^x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2L
$conv1d_1/ExpandDims_1/ReadVariableOp$conv1d_1/ExpandDims_1/ReadVariableOp2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp2j
3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w1/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w1/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_89277S
<x_deep_fm_1_cin_1_w0_regularizer_abs_readvariableop_resource:? 
identity??3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
&x_deep_fm_1/cin_1/w0/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&x_deep_fm_1/cin_1/w0/Regularizer/Const?
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOpReadVariableOp<x_deep_fm_1_cin_1_w0_regularizer_abs_readvariableop_resource*#
_output_shapes
:? *
dtype025
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp?
$x_deep_fm_1/cin_1/w0/Regularizer/AbsAbs;x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Abs?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_1?
$x_deep_fm_1/cin_1/w0/Regularizer/SumSum(x_deep_fm_1/cin_1/w0/Regularizer/Abs:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/Sum?
&x_deep_fm_1/cin_1/w0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??'72(
&x_deep_fm_1/cin_1/w0/Regularizer/mul/x?
$x_deep_fm_1/cin_1/w0/Regularizer/mulMul/x_deep_fm_1/cin_1/w0/Regularizer/mul/x:output:0-x_deep_fm_1/cin_1/w0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/mul?
$x_deep_fm_1/cin_1/w0/Regularizer/addAddV2/x_deep_fm_1/cin_1/w0/Regularizer/Const:output:0(x_deep_fm_1/cin_1/w0/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$x_deep_fm_1/cin_1/w0/Regularizer/add?
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOpReadVariableOp<x_deep_fm_1_cin_1_w0_regularizer_abs_readvariableop_resource*#
_output_shapes
:? *
dtype028
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp?
'x_deep_fm_1/cin_1/w0/Regularizer/SquareSquare>x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp:value:0*
T0*#
_output_shapes
:? 2)
'x_deep_fm_1/cin_1/w0/Regularizer/Square?
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(x_deep_fm_1/cin_1/w0/Regularizer/Const_2?
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1Sum+x_deep_fm_1/cin_1/w0/Regularizer/Square:y:01x_deep_fm_1/cin_1/w0/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/Sum_1?
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2*
(x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x?
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1Mul1x_deep_fm_1/cin_1/w0/Regularizer/mul_1/x:output:0/x_deep_fm_1/cin_1/w0/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/mul_1?
&x_deep_fm_1/cin_1/w0/Regularizer/add_1AddV2(x_deep_fm_1/cin_1/w0/Regularizer/add:z:0*x_deep_fm_1/cin_1/w0/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2(
&x_deep_fm_1/cin_1/w0/Regularizer/add_1t
IdentityIdentity*x_deep_fm_1/cin_1/w0/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp4^x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp7^x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp3x_deep_fm_1/cin_1/w0/Regularizer/Abs/ReadVariableOp2p
6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp6x_deep_fm_1/cin_1/w0/Regularizer/Square/ReadVariableOp
?
?
,__inference_embedding_33_layer_call_fn_89423

inputs
unknown:C
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_855012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_51_layer_call_fn_89729

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_858252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_34_layer_call_fn_89440

inputs
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_855192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_48_layer_call_and_return_conditional_losses_85771

inputs(
embedding_lookup_85765:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85765Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85765*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85765*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_9_layer_call_and_return_conditional_losses_89257

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?)
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85243

inputs5
'assignmovingavg_readvariableop_resource:'7
)assignmovingavg_1_readvariableop_resource:'*
cast_readvariableop_resource:',
cast_1_readvariableop_resource:'
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?Cast/ReadVariableOp?Cast_1/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:'2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:?????????'2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:'*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:'*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:'*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:'*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:'2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:'2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:'2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:'2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:'2
batchnorm/mul_2|
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_44_layer_call_and_return_conditional_losses_85699

inputs)
embedding_lookup_85693:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85693Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85693*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85693*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_30_layer_call_and_return_conditional_losses_85447

inputs(
embedding_lookup_85441:*
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85441Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85441*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85441*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_linear_1_layer_call_fn_88880

inputs
unknown:'
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_linear_1_layer_call_and_return_conditional_losses_853552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????': : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_51_layer_call_and_return_conditional_losses_85825

inputs)
embedding_lookup_85819:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85819Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85819*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85819*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85183

inputs*
cast_readvariableop_resource:',
cast_1_readvariableop_resource:',
cast_2_readvariableop_resource:',
cast_3_readvariableop_resource:'
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOp?Cast_2/ReadVariableOp?Cast_3/ReadVariableOp?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast/ReadVariableOp?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_1/ReadVariableOp?
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_2/ReadVariableOp?
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
:'*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:'2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:'2
batchnorm/Rsqrt~
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:'2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/mul_1~
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:'2
batchnorm/mul_2~
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:'2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:?????????'2
batchnorm/add_1n
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:?????????'2

Identity?
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????': : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:O K
'
_output_shapes
:?????????'
 
_user_specified_nameinputs
?

?
G__inference_embedding_41_layer_call_and_return_conditional_losses_89569

inputs)
embedding_lookup_89563:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89563Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89563*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89563*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_34_layer_call_and_return_conditional_losses_89450

inputs(
embedding_lookup_89444:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89444Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89444*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89444*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_35_layer_call_and_return_conditional_losses_89467

inputs)
embedding_lookup_89461:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89461Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89461*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89461*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_39_layer_call_and_return_conditional_losses_89535

inputs(
embedding_lookup_89529:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89529Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89529*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89529*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_48_layer_call_and_return_conditional_losses_89688

inputs(
embedding_lookup_89682:
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_89682Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/89682*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/89682*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
G__inference_embedding_41_layer_call_and_return_conditional_losses_85645

inputs)
embedding_lookup_85639:	?
identity??embedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_85639Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/85639*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/85639*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88992

inputs9
&dense_6_matmul_readvariableop_resource:	?@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@@5
'dense_7_biasadd_readvariableop_resource:@8
&dense_8_matmul_readvariableop_resource:@5
'dense_8_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_6/Relu?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_7/BiasAddp
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_7/Relu?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????'<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:½
?
dense_feature_columns
sparse_feature_columns
embed_layers
bn

linear
dense_layer
	cin_layer
	out_layer
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_model
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
?
0
1
2
3
 4
!5
"6
#7
$8
%9
&10
'11
(12
)13
*14
+15
,16
-17
.18
/19
020
121
222
323
424
525"
trackable_list_wrapper
?
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13
D14
E15
F16
G17
H18
I19
J20
K21
L22
M23
N24
O25"
trackable_list_wrapper
?
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Y	out_layer
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^hidden_layers
_	out_layer
`dropout
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
ecin_size
f	field_num
gw0
hw1
	icin_W
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

nkernel
obias
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rateQm?Rm?gm?hm?nm?om?ym?zm?{m?|m?}m?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Qv?Rv?gv?hv?nv?ov?yv?zv?{v?|v?}v?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
y0
z1
{2
|3
}4
~5
6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
Q26
R27
S28
T29
?30
?31
?32
?33
?34
?35
?36
?37
g38
h39
n40
o41"
trackable_list_wrapper
?
y0
z1
{2
|3
}4
~5
6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
Q26
R27
?28
?29
?30
?31
?32
?33
?34
?35
g36
h37
n38
o39"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?

	variables
trainable_variables
regularization_losses
 ?layer_regularization_losses
?layer_metrics
?layers
?metrics
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
?
y
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
z
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
{
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
|
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
}
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
~
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?
embeddings
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5:3'2'x_deep_fm_1/batch_normalization_1/gamma
4:2'2&x_deep_fm_1/batch_normalization_1/beta
=:;' (2-x_deep_fm_1/batch_normalization_1/moving_mean
A:?' (21x_deep_fm_1/batch_normalization_1/moving_variance
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
U	variables
Vtrainable_variables
Wregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Z	variables
[trainable_variables
\regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
a	variables
btrainable_variables
cregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
+:)? 2x_deep_fm_1/cin_1/w0
+:)? 2x_deep_fm_1/cin_1/w1
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
j	variables
ktrainable_variables
lregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*@2x_deep_fm_1/dense_9/kernel
&:$2x_deep_fm_1/dense_9/bias
.
n0
o1"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
p	variables
qtrainable_variables
rregularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5:3h2#x_deep_fm_1/embedding_26/embeddings
6:4	?2#x_deep_fm_1/embedding_27/embeddings
6:4	?2#x_deep_fm_1/embedding_28/embeddings
6:4	?2#x_deep_fm_1/embedding_29/embeddings
5:3*2#x_deep_fm_1/embedding_30/embeddings
5:3	2#x_deep_fm_1/embedding_31/embeddings
6:4	?2#x_deep_fm_1/embedding_32/embeddings
5:3C2#x_deep_fm_1/embedding_33/embeddings
5:32#x_deep_fm_1/embedding_34/embeddings
6:4	?2#x_deep_fm_1/embedding_35/embeddings
6:4	?2#x_deep_fm_1/embedding_36/embeddings
6:4	?2#x_deep_fm_1/embedding_37/embeddings
6:4	?2#x_deep_fm_1/embedding_38/embeddings
5:32#x_deep_fm_1/embedding_39/embeddings
6:4	?
2#x_deep_fm_1/embedding_40/embeddings
6:4	?2#x_deep_fm_1/embedding_41/embeddings
5:3
2#x_deep_fm_1/embedding_42/embeddings
6:4	?2#x_deep_fm_1/embedding_43/embeddings
6:4	?2#x_deep_fm_1/embedding_44/embeddings
5:32#x_deep_fm_1/embedding_45/embeddings
6:4	?2#x_deep_fm_1/embedding_46/embeddings
5:3
2#x_deep_fm_1/embedding_47/embeddings
5:32#x_deep_fm_1/embedding_48/embeddings
6:4	?2#x_deep_fm_1/embedding_49/embeddings
5:3'2#x_deep_fm_1/embedding_50/embeddings
6:4	?2#x_deep_fm_1/embedding_51/embeddings
5:3'2#x_deep_fm_1/linear_1/dense_5/kernel
/:-2!x_deep_fm_1/linear_1/dense_5/bias
;:9	?@2(x_deep_fm_1/dense_layer_1/dense_6/kernel
4:2@2&x_deep_fm_1/dense_layer_1/dense_6/bias
::8@@2(x_deep_fm_1/dense_layer_1/dense_7/kernel
4:2@2&x_deep_fm_1/dense_layer_1/dense_7/bias
::8@2(x_deep_fm_1/dense_layer_1/dense_8/kernel
4:22&x_deep_fm_1/dense_layer_1/dense_8/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13
D14
E15
F16
G17
H18
I19
J20
K21
L22
M23
N24
O25
26
27
28
29
30"
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
z0"
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
{0"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
|0"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
}0"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
S0
T1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Y0"
trackable_list_wrapper
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
>
?0
?1
_2
`3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
?
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
v
?
thresholds
?true_positives
?false_negatives
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?trainable_variables
?regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 ?layer_regularization_losses
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
::8'2.Adam/x_deep_fm_1/batch_normalization_1/gamma/m
9:7'2-Adam/x_deep_fm_1/batch_normalization_1/beta/m
0:.? 2Adam/x_deep_fm_1/cin_1/w0/m
0:.? 2Adam/x_deep_fm_1/cin_1/w1/m
1:/@2!Adam/x_deep_fm_1/dense_9/kernel/m
+:)2Adam/x_deep_fm_1/dense_9/bias/m
::8h2*Adam/x_deep_fm_1/embedding_26/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_27/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_28/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_29/embeddings/m
::8*2*Adam/x_deep_fm_1/embedding_30/embeddings/m
::8	2*Adam/x_deep_fm_1/embedding_31/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_32/embeddings/m
::8C2*Adam/x_deep_fm_1/embedding_33/embeddings/m
::82*Adam/x_deep_fm_1/embedding_34/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_35/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_36/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_37/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_38/embeddings/m
::82*Adam/x_deep_fm_1/embedding_39/embeddings/m
;:9	?
2*Adam/x_deep_fm_1/embedding_40/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_41/embeddings/m
::8
2*Adam/x_deep_fm_1/embedding_42/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_43/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_44/embeddings/m
::82*Adam/x_deep_fm_1/embedding_45/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_46/embeddings/m
::8
2*Adam/x_deep_fm_1/embedding_47/embeddings/m
::82*Adam/x_deep_fm_1/embedding_48/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_49/embeddings/m
::8'2*Adam/x_deep_fm_1/embedding_50/embeddings/m
;:9	?2*Adam/x_deep_fm_1/embedding_51/embeddings/m
::8'2*Adam/x_deep_fm_1/linear_1/dense_5/kernel/m
4:22(Adam/x_deep_fm_1/linear_1/dense_5/bias/m
@:>	?@2/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/m
9:7@2-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/m
?:=@@2/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/m
9:7@2-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/m
?:=@2/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/m
9:72-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/m
::8'2.Adam/x_deep_fm_1/batch_normalization_1/gamma/v
9:7'2-Adam/x_deep_fm_1/batch_normalization_1/beta/v
0:.? 2Adam/x_deep_fm_1/cin_1/w0/v
0:.? 2Adam/x_deep_fm_1/cin_1/w1/v
1:/@2!Adam/x_deep_fm_1/dense_9/kernel/v
+:)2Adam/x_deep_fm_1/dense_9/bias/v
::8h2*Adam/x_deep_fm_1/embedding_26/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_27/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_28/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_29/embeddings/v
::8*2*Adam/x_deep_fm_1/embedding_30/embeddings/v
::8	2*Adam/x_deep_fm_1/embedding_31/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_32/embeddings/v
::8C2*Adam/x_deep_fm_1/embedding_33/embeddings/v
::82*Adam/x_deep_fm_1/embedding_34/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_35/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_36/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_37/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_38/embeddings/v
::82*Adam/x_deep_fm_1/embedding_39/embeddings/v
;:9	?
2*Adam/x_deep_fm_1/embedding_40/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_41/embeddings/v
::8
2*Adam/x_deep_fm_1/embedding_42/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_43/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_44/embeddings/v
::82*Adam/x_deep_fm_1/embedding_45/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_46/embeddings/v
::8
2*Adam/x_deep_fm_1/embedding_47/embeddings/v
::82*Adam/x_deep_fm_1/embedding_48/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_49/embeddings/v
::8'2*Adam/x_deep_fm_1/embedding_50/embeddings/v
;:9	?2*Adam/x_deep_fm_1/embedding_51/embeddings/v
::8'2*Adam/x_deep_fm_1/linear_1/dense_5/kernel/v
4:22(Adam/x_deep_fm_1/linear_1/dense_5/bias/v
@:>	?@2/Adam/x_deep_fm_1/dense_layer_1/dense_6/kernel/v
9:7@2-Adam/x_deep_fm_1/dense_layer_1/dense_6/bias/v
?:=@@2/Adam/x_deep_fm_1/dense_layer_1/dense_7/kernel/v
9:7@2-Adam/x_deep_fm_1/dense_layer_1/dense_7/bias/v
?:=@2/Adam/x_deep_fm_1/dense_layer_1/dense_8/kernel/v
9:72-Adam/x_deep_fm_1/dense_layer_1/dense_8/bias/v
?2?
+__inference_x_deep_fm_1_layer_call_fn_86117
+__inference_x_deep_fm_1_layer_call_fn_87837
+__inference_x_deep_fm_1_layer_call_fn_87926
+__inference_x_deep_fm_1_layer_call_fn_87079?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_85159input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88352
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88791
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87350
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87621?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_1_layer_call_fn_88804
5__inference_batch_normalization_1_layer_call_fn_88817?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88837
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88871?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_linear_1_layer_call_fn_88880
(__inference_linear_1_layer_call_fn_88889?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
C__inference_linear_1_layer_call_and_return_conditional_losses_88899
C__inference_linear_1_layer_call_and_return_conditional_losses_88909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
-__inference_dense_layer_1_layer_call_fn_88926
-__inference_dense_layer_1_layer_call_fn_88943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88968
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88992?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
%__inference_cin_1_layer_call_fn_89031
%__inference_cin_1_layer_call_fn_89040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
@__inference_cin_1_layer_call_and_return_conditional_losses_89139
@__inference_cin_1_layer_call_and_return_conditional_losses_89238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
'__inference_dense_9_layer_call_fn_89247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_9_layer_call_and_return_conditional_losses_89257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_89277?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_89297?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_87748input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_26_layer_call_fn_89304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_26_layer_call_and_return_conditional_losses_89314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_27_layer_call_fn_89321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_27_layer_call_and_return_conditional_losses_89331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_28_layer_call_fn_89338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_28_layer_call_and_return_conditional_losses_89348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_29_layer_call_fn_89355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_29_layer_call_and_return_conditional_losses_89365?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_30_layer_call_fn_89372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_30_layer_call_and_return_conditional_losses_89382?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_31_layer_call_fn_89389?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_31_layer_call_and_return_conditional_losses_89399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_32_layer_call_fn_89406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_32_layer_call_and_return_conditional_losses_89416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_33_layer_call_fn_89423?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_33_layer_call_and_return_conditional_losses_89433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_34_layer_call_fn_89440?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_34_layer_call_and_return_conditional_losses_89450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_35_layer_call_fn_89457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_35_layer_call_and_return_conditional_losses_89467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_36_layer_call_fn_89474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_36_layer_call_and_return_conditional_losses_89484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_37_layer_call_fn_89491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_37_layer_call_and_return_conditional_losses_89501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_38_layer_call_fn_89508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_38_layer_call_and_return_conditional_losses_89518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_39_layer_call_fn_89525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_39_layer_call_and_return_conditional_losses_89535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_40_layer_call_fn_89542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_40_layer_call_and_return_conditional_losses_89552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_41_layer_call_fn_89559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_41_layer_call_and_return_conditional_losses_89569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_42_layer_call_fn_89576?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_42_layer_call_and_return_conditional_losses_89586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_43_layer_call_fn_89593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_43_layer_call_and_return_conditional_losses_89603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_44_layer_call_fn_89610?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_44_layer_call_and_return_conditional_losses_89620?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_45_layer_call_fn_89627?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_45_layer_call_and_return_conditional_losses_89637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_46_layer_call_fn_89644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_46_layer_call_and_return_conditional_losses_89654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_47_layer_call_fn_89661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_47_layer_call_and_return_conditional_losses_89671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_48_layer_call_fn_89678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_48_layer_call_and_return_conditional_losses_89688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_49_layer_call_fn_89695?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_49_layer_call_and_return_conditional_losses_89705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_50_layer_call_fn_89712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_50_layer_call_and_return_conditional_losses_89722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_51_layer_call_fn_89729?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_51_layer_call_and_return_conditional_losses_89739?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_85159?ESTRQ??yz{|}~???????????????????gh??????no0?-
&?#
!?
input_1?????????'
? "3?0
.
output_1"?
output_1??????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88837bSTRQ3?0
)?&
 ?
inputs?????????'
p 
? "%?"
?
0?????????'
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_88871bSTRQ3?0
)?&
 ?
inputs?????????'
p
? "%?"
?
0?????????'
? ?
5__inference_batch_normalization_1_layer_call_fn_88804USTRQ3?0
)?&
 ?
inputs?????????'
p 
? "??????????'?
5__inference_batch_normalization_1_layer_call_fn_88817USTRQ3?0
)?&
 ?
inputs?????????'
p
? "??????????'?
@__inference_cin_1_layer_call_and_return_conditional_losses_89139pghC?@
)?&
$?!
inputs?????????
?

trainingp "%?"
?
0?????????@
? ?
@__inference_cin_1_layer_call_and_return_conditional_losses_89238pghC?@
)?&
$?!
inputs?????????
?

trainingp"%?"
?
0?????????@
? ?
%__inference_cin_1_layer_call_fn_89031cghC?@
)?&
$?!
inputs?????????
?

trainingp "??????????@?
%__inference_cin_1_layer_call_fn_89040cghC?@
)?&
$?!
inputs?????????
?

trainingp"??????????@?
B__inference_dense_9_layer_call_and_return_conditional_losses_89257\no/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_9_layer_call_fn_89247Ono/?,
%?"
 ?
inputs?????????@
? "???????????
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88968w??????@?=
&?#
!?
inputs??????????
?

trainingp "%?"
?
0?????????
? ?
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_88992w??????@?=
&?#
!?
inputs??????????
?

trainingp"%?"
?
0?????????
? ?
-__inference_dense_layer_1_layer_call_fn_88926j??????@?=
&?#
!?
inputs??????????
?

trainingp "???????????
-__inference_dense_layer_1_layer_call_fn_88943j??????@?=
&?#
!?
inputs??????????
?

trainingp"???????????
G__inference_embedding_26_layer_call_and_return_conditional_losses_89314Wy+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_26_layer_call_fn_89304Jy+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_27_layer_call_and_return_conditional_losses_89331Wz+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_27_layer_call_fn_89321Jz+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_28_layer_call_and_return_conditional_losses_89348W{+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_28_layer_call_fn_89338J{+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_29_layer_call_and_return_conditional_losses_89365W|+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_29_layer_call_fn_89355J|+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_30_layer_call_and_return_conditional_losses_89382W}+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_30_layer_call_fn_89372J}+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_31_layer_call_and_return_conditional_losses_89399W~+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_31_layer_call_fn_89389J~+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_32_layer_call_and_return_conditional_losses_89416W+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? z
,__inference_embedding_32_layer_call_fn_89406J+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_33_layer_call_and_return_conditional_losses_89433X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_33_layer_call_fn_89423K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_34_layer_call_and_return_conditional_losses_89450X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_34_layer_call_fn_89440K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_35_layer_call_and_return_conditional_losses_89467X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_35_layer_call_fn_89457K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_36_layer_call_and_return_conditional_losses_89484X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_36_layer_call_fn_89474K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_37_layer_call_and_return_conditional_losses_89501X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_37_layer_call_fn_89491K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_38_layer_call_and_return_conditional_losses_89518X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_38_layer_call_fn_89508K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_39_layer_call_and_return_conditional_losses_89535X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_39_layer_call_fn_89525K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_40_layer_call_and_return_conditional_losses_89552X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_40_layer_call_fn_89542K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_41_layer_call_and_return_conditional_losses_89569X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_41_layer_call_fn_89559K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_42_layer_call_and_return_conditional_losses_89586X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_42_layer_call_fn_89576K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_43_layer_call_and_return_conditional_losses_89603X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_43_layer_call_fn_89593K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_44_layer_call_and_return_conditional_losses_89620X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_44_layer_call_fn_89610K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_45_layer_call_and_return_conditional_losses_89637X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_45_layer_call_fn_89627K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_46_layer_call_and_return_conditional_losses_89654X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_46_layer_call_fn_89644K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_47_layer_call_and_return_conditional_losses_89671X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_47_layer_call_fn_89661K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_48_layer_call_and_return_conditional_losses_89688X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_48_layer_call_fn_89678K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_49_layer_call_and_return_conditional_losses_89705X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_49_layer_call_fn_89695K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_50_layer_call_and_return_conditional_losses_89722X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_50_layer_call_fn_89712K?+?(
!?
?
inputs?????????
? "???????????
G__inference_embedding_51_layer_call_and_return_conditional_losses_89739X?+?(
!?
?
inputs?????????
? "%?"
?
0?????????
? {
,__inference_embedding_51_layer_call_fn_89729K?+?(
!?
?
inputs?????????
? "???????????
C__inference_linear_1_layer_call_and_return_conditional_losses_88899n????<
%?"
 ?
inputs?????????'
?

trainingp "%?"
?
0?????????
? ?
C__inference_linear_1_layer_call_and_return_conditional_losses_88909n????<
%?"
 ?
inputs?????????'
?

trainingp"%?"
?
0?????????
? ?
(__inference_linear_1_layer_call_fn_88880a????<
%?"
 ?
inputs?????????'
?

trainingp "???????????
(__inference_linear_1_layer_call_fn_88889a????<
%?"
 ?
inputs?????????'
?

trainingp"??????????:
__inference_loss_fn_0_89277g?

? 
? "? :
__inference_loss_fn_1_89297h?

? 
? "? ?
#__inference_signature_wrapper_87748?ESTRQ??yz{|}~???????????????????gh??????no;?8
? 
1?.
,
input_1!?
input_1?????????'"3?0
.
output_1"?
output_1??????????
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87350?ESTRQ??yz{|}~???????????????????gh??????no8?5
.?+
!?
input_1?????????'
p 

 
? "%?"
?
0?????????
? ?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_87621?ESTRQ??yz{|}~???????????????????gh??????no8?5
.?+
!?
input_1?????????'
p

 
? "%?"
?
0?????????
? ?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88352?ESTRQ??yz{|}~???????????????????gh??????no7?4
-?*
 ?
inputs?????????'
p 

 
? "%?"
?
0?????????
? ?
F__inference_x_deep_fm_1_layer_call_and_return_conditional_losses_88791?ESTRQ??yz{|}~???????????????????gh??????no7?4
-?*
 ?
inputs?????????'
p

 
? "%?"
?
0?????????
? ?
+__inference_x_deep_fm_1_layer_call_fn_86117?ESTRQ??yz{|}~???????????????????gh??????no8?5
.?+
!?
input_1?????????'
p 

 
? "???????????
+__inference_x_deep_fm_1_layer_call_fn_87079?ESTRQ??yz{|}~???????????????????gh??????no8?5
.?+
!?
input_1?????????'
p

 
? "???????????
+__inference_x_deep_fm_1_layer_call_fn_87837?ESTRQ??yz{|}~???????????????????gh??????no7?4
-?*
 ?
inputs?????????'
p 

 
? "???????????
+__inference_x_deep_fm_1_layer_call_fn_87926?ESTRQ??yz{|}~???????????????????gh??????no7?4
-?*
 ?
inputs?????????'
p

 
? "??????????