Є╦:
╩Ы
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
,
Exp
x"T
y"T"
Ttype:

2
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8└√5
═
:dcn__attention__parallel__v1_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*K
shared_name<:dcn__attention__parallel__v1_1/batch_normalization_1/gamma
╞
Ndcn__attention__parallel__v1_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp:dcn__attention__parallel__v1_1/batch_normalization_1/gamma*
_output_shapes	
:▌*
dtype0
╦
9dcn__attention__parallel__v1_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*J
shared_name;9dcn__attention__parallel__v1_1/batch_normalization_1/beta
─
Mdcn__attention__parallel__v1_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp9dcn__attention__parallel__v1_1/batch_normalization_1/beta*
_output_shapes	
:▌*
dtype0
┘
@dcn__attention__parallel__v1_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*Q
shared_nameB@dcn__attention__parallel__v1_1/batch_normalization_1/moving_mean
╥
Tdcn__attention__parallel__v1_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp@dcn__attention__parallel__v1_1/batch_normalization_1/moving_mean*
_output_shapes	
:▌*
dtype0
с
Ddcn__attention__parallel__v1_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*U
shared_nameFDdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance
┌
Xdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOpDdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance*
_output_shapes	
:▌*
dtype0
╜
0dcn__attention__parallel__v1_1/cross_layer_1/wq0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*A
shared_name20dcn__attention__parallel__v1_1/cross_layer_1/wq0
╢
Ddcn__attention__parallel__v1_1/cross_layer_1/wq0/Read/ReadVariableOpReadVariableOp0dcn__attention__parallel__v1_1/cross_layer_1/wq0*
_output_shapes
:	▌*
dtype0
╜
0dcn__attention__parallel__v1_1/cross_layer_1/wk0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*A
shared_name20dcn__attention__parallel__v1_1/cross_layer_1/wk0
╢
Ddcn__attention__parallel__v1_1/cross_layer_1/wk0/Read/ReadVariableOpReadVariableOp0dcn__attention__parallel__v1_1/cross_layer_1/wk0*
_output_shapes
:	▌*
dtype0
╜
0dcn__attention__parallel__v1_1/cross_layer_1/wv0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*A
shared_name20dcn__attention__parallel__v1_1/cross_layer_1/wv0
╢
Ddcn__attention__parallel__v1_1/cross_layer_1/wv0/Read/ReadVariableOpReadVariableOp0dcn__attention__parallel__v1_1/cross_layer_1/wv0*
_output_shapes
:	▌*
dtype0
╗
/dcn__attention__parallel__v1_1/cross_layer_1/b0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*@
shared_name1/dcn__attention__parallel__v1_1/cross_layer_1/b0
┤
Cdcn__attention__parallel__v1_1/cross_layer_1/b0/Read/ReadVariableOpReadVariableOp/dcn__attention__parallel__v1_1/cross_layer_1/b0*
_output_shapes
:	▌*
dtype0
╖
-dcn__attention__parallel__v1_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▐*>
shared_name/-dcn__attention__parallel__v1_1/dense_7/kernel
░
Adcn__attention__parallel__v1_1/dense_7/kernel/Read/ReadVariableOpReadVariableOp-dcn__attention__parallel__v1_1/dense_7/kernel*
_output_shapes
:	▐*
dtype0
о
+dcn__attention__parallel__v1_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+dcn__attention__parallel__v1_1/dense_7/bias
з
?dcn__attention__parallel__v1_1/dense_7/bias/Read/ReadVariableOpReadVariableOp+dcn__attention__parallel__v1_1/dense_7/bias*
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
╚
6dcn__attention__parallel__v1_1/embedding_26/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*G
shared_name86dcn__attention__parallel__v1_1/embedding_26/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_26/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_26/embeddings*
_output_shapes

:h*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_27/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *G
shared_name86dcn__attention__parallel__v1_1/embedding_27/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_27/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_27/embeddings*
_output_shapes
:	 *
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_36/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Т*G
shared_name86dcn__attention__parallel__v1_1/embedding_36/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_36/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_36/embeddings*
_output_shapes
:	Т*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_37/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	х*G
shared_name86dcn__attention__parallel__v1_1/embedding_37/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_37/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_37/embeddings*
_output_shapes
:	х*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_38/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┌*G
shared_name86dcn__attention__parallel__v1_1/embedding_38/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_38/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_38/embeddings*
_output_shapes
:	┌*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_39/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86dcn__attention__parallel__v1_1/embedding_39/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_39/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_39/embeddings*
_output_shapes

:*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_40/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪
*G
shared_name86dcn__attention__parallel__v1_1/embedding_40/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_40/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_40/embeddings*
_output_shapes
:	╪
*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_41/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╔*G
shared_name86dcn__attention__parallel__v1_1/embedding_41/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_41/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_41/embeddings*
_output_shapes
:	╔*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_42/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*G
shared_name86dcn__attention__parallel__v1_1/embedding_42/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_42/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_42/embeddings*
_output_shapes

:
*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_43/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Г*G
shared_name86dcn__attention__parallel__v1_1/embedding_43/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_43/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_43/embeddings*
_output_shapes
:	Г*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_44/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*G
shared_name86dcn__attention__parallel__v1_1/embedding_44/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_44/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_44/embeddings*
_output_shapes
:	└*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_45/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86dcn__attention__parallel__v1_1/embedding_45/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_45/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_45/embeddings*
_output_shapes

:*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_28/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╬*G
shared_name86dcn__attention__parallel__v1_1/embedding_28/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_28/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_28/embeddings*
_output_shapes
:	╬*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_46/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	З*G
shared_name86dcn__attention__parallel__v1_1/embedding_46/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_46/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_46/embeddings*
_output_shapes
:	З*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_47/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*G
shared_name86dcn__attention__parallel__v1_1/embedding_47/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_47/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_47/embeddings*
_output_shapes

:
*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_48/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86dcn__attention__parallel__v1_1/embedding_48/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_48/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_48/embeddings*
_output_shapes

:*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_49/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*G
shared_name86dcn__attention__parallel__v1_1/embedding_49/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_49/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_49/embeddings*
_output_shapes
:	╕*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_50/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*G
shared_name86dcn__attention__parallel__v1_1/embedding_50/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_50/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_50/embeddings*
_output_shapes

:'*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_51/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╩*G
shared_name86dcn__attention__parallel__v1_1/embedding_51/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_51/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_51/embeddings*
_output_shapes
:	╩*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_29/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┼*G
shared_name86dcn__attention__parallel__v1_1/embedding_29/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_29/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_29/embeddings*
_output_shapes
:	┼*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_30/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**G
shared_name86dcn__attention__parallel__v1_1/embedding_30/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_30/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_30/embeddings*
_output_shapes

:**
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_31/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*G
shared_name86dcn__attention__parallel__v1_1/embedding_31/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_31/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_31/embeddings*
_output_shapes

:	*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_32/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╨*G
shared_name86dcn__attention__parallel__v1_1/embedding_32/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_32/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_32/embeddings*
_output_shapes
:	╨*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_33/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*G
shared_name86dcn__attention__parallel__v1_1/embedding_33/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_33/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_33/embeddings*
_output_shapes

:C*
dtype0
╚
6dcn__attention__parallel__v1_1/embedding_34/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86dcn__attention__parallel__v1_1/embedding_34/embeddings
┴
Jdcn__attention__parallel__v1_1/embedding_34/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_34/embeddings*
_output_shapes

:*
dtype0
╔
6dcn__attention__parallel__v1_1/embedding_35/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*G
shared_name86dcn__attention__parallel__v1_1/embedding_35/embeddings
┬
Jdcn__attention__parallel__v1_1/embedding_35/embeddings/Read/ReadVariableOpReadVariableOp6dcn__attention__parallel__v1_1/embedding_35/embeddings*
_output_shapes
:	╕*
dtype0
╙
;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌ *L
shared_name=;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel
╠
Odcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/Read/ReadVariableOpReadVariableOp;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel*
_output_shapes
:	▌ *
dtype0
╩
9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias
├
Mdcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/Read/ReadVariableOpReadVariableOp9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias*
_output_shapes
: *
dtype0
╥
;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *L
shared_name=;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel
╦
Odcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/Read/ReadVariableOpReadVariableOp;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel*
_output_shapes

:  *
dtype0
╩
9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias
├
Mdcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/Read/ReadVariableOpReadVariableOp9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias*
_output_shapes
: *
dtype0
╥
;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *L
shared_name=;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel
╦
Odcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/Read/ReadVariableOpReadVariableOp;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel*
_output_shapes

: *
dtype0
╩
9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias
├
Mdcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/Read/ReadVariableOpReadVariableOp9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias*
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
shape:╚*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:╚*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:╚*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:╚*
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
█
AAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*R
shared_nameCAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m
╘
UAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOpAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m*
_output_shapes	
:▌*
dtype0
┘
@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m
╥
TAdam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m*
_output_shapes	
:▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m*
_output_shapes
:	▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m*
_output_shapes
:	▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m*
_output_shapes
:	▌*
dtype0
╔
6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*G
shared_name86Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m
┬
JAdam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m/Read/ReadVariableOpReadVariableOp6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m*
_output_shapes
:	▌*
dtype0
┼
4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▐*E
shared_name64Adam/dcn__attention__parallel__v1_1/dense_7/kernel/m
╛
HAdam/dcn__attention__parallel__v1_1/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/m*
_output_shapes
:	▐*
dtype0
╝
2Adam/dcn__attention__parallel__v1_1/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/dcn__attention__parallel__v1_1/dense_7/bias/m
╡
FAdam/dcn__attention__parallel__v1_1/dense_7/bias/m/Read/ReadVariableOpReadVariableOp2Adam/dcn__attention__parallel__v1_1/dense_7/bias/m*
_output_shapes
:*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m*
_output_shapes

:h*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m*
_output_shapes
:	 *
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Т*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m*
_output_shapes
:	Т*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	х*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m*
_output_shapes
:	х*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┌*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m*
_output_shapes
:	┌*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m*
_output_shapes
:	╪
*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╔*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m*
_output_shapes
:	╔*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m*
_output_shapes

:
*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Г*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m*
_output_shapes
:	Г*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m*
_output_shapes
:	└*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╬*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m*
_output_shapes
:	╬*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	З*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m*
_output_shapes
:	З*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m*
_output_shapes

:
*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m*
_output_shapes
:	╕*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m*
_output_shapes

:'*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╩*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m*
_output_shapes
:	╩*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┼*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m*
_output_shapes
:	┼*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m*
_output_shapes

:**
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m*
_output_shapes

:	*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╨*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m*
_output_shapes
:	╨*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m*
_output_shapes

:C*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m
╧
QAdam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/m
╨
QAdam/dcn__attention__parallel__v1_1/embedding_35/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/m*
_output_shapes
:	╕*
dtype0
с
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌ *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m
┌
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m*
_output_shapes
:	▌ *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/m
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/m/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/m*
_output_shapes
: *
dtype0
р
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m
┘
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m*
_output_shapes

:  *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/m
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/m/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/m*
_output_shapes
: *
dtype0
р
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m
┘
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m*
_output_shapes

: *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/m
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/m/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/m*
_output_shapes
:*
dtype0
█
AAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*R
shared_nameCAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v
╘
UAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOpAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v*
_output_shapes	
:▌*
dtype0
┘
@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:▌*Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v
╥
TAdam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v*
_output_shapes	
:▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v*
_output_shapes
:	▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v*
_output_shapes
:	▌*
dtype0
╦
7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*H
shared_name97Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v
─
KAdam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v/Read/ReadVariableOpReadVariableOp7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v*
_output_shapes
:	▌*
dtype0
╔
6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌*G
shared_name86Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v
┬
JAdam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v/Read/ReadVariableOpReadVariableOp6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v*
_output_shapes
:	▌*
dtype0
┼
4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▐*E
shared_name64Adam/dcn__attention__parallel__v1_1/dense_7/kernel/v
╛
HAdam/dcn__attention__parallel__v1_1/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/v*
_output_shapes
:	▐*
dtype0
╝
2Adam/dcn__attention__parallel__v1_1/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adam/dcn__attention__parallel__v1_1/dense_7/bias/v
╡
FAdam/dcn__attention__parallel__v1_1/dense_7/bias/v/Read/ReadVariableOpReadVariableOp2Adam/dcn__attention__parallel__v1_1/dense_7/bias/v*
_output_shapes
:*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:h*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v*
_output_shapes

:h*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 *N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v*
_output_shapes
:	 *
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Т*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v*
_output_shapes
:	Т*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	х*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v*
_output_shapes
:	х*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┌*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v*
_output_shapes
:	┌*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╪
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v*
_output_shapes
:	╪
*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╔*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v*
_output_shapes
:	╔*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v*
_output_shapes

:
*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Г*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v*
_output_shapes
:	Г*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	└*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v*
_output_shapes
:	└*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╬*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v*
_output_shapes
:	╬*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	З*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v*
_output_shapes
:	З*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v*
_output_shapes

:
*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v*
_output_shapes
:	╕*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:'*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v*
_output_shapes

:'*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╩*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v*
_output_shapes
:	╩*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	┼*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v*
_output_shapes
:	┼*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v*
_output_shapes

:**
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v*
_output_shapes

:	*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╨*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v*
_output_shapes
:	╨*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:C*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v*
_output_shapes

:C*
dtype0
╓
=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v
╧
QAdam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v*
_output_shapes

:*
dtype0
╫
=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╕*N
shared_name?=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/v
╨
QAdam/dcn__attention__parallel__v1_1/embedding_35/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/v*
_output_shapes
:	╕*
dtype0
с
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	▌ *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v
┌
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v*
_output_shapes
:	▌ *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/v
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/v/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/v*
_output_shapes
: *
dtype0
р
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v
┘
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v*
_output_shapes

:  *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/v
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/v/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/v*
_output_shapes
: *
dtype0
р
BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *S
shared_nameDBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v
┘
VAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v*
_output_shapes

: *
dtype0
╪
@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*Q
shared_nameB@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v
╤
TAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v/Read/ReadVariableOpReadVariableOp@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ЧБ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╤А
value╞АB┬А B║А
Ў
dense_feature_columns
sparse_feature_columns
embed_layers
bn
dense_layer
cross_layer
output_layer
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
^
0
1
2
3
4
5
6
7
8
9
10
11
12
╞
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
.19
/20
021
122
223
324
425
т
5embed_0
6embed_1
7embed_2
8embed_3
9embed_4
:embed_5
;embed_6
<embed_7
=embed_8
>embed_9
?embed_10
@embed_11
Aembed_12
Bembed_13
Cembed_14
Dembed_15
Eembed_16
Fembed_17
Gembed_18
Hembed_19
Iembed_20
Jembed_21
Kembed_22
Lembed_23
Membed_24
Nembed_25
Ч
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
v
Xhidden_layer
Youtput_layer
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
Э
^wq0
_wq
`wk0
awk
bwv0
cwv
db0
e
cross_bias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
К
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratePm╟Qm╚^m╔`m╩bm╦dm╠jm═km╬um╧vm╨wm╤xm╥ym╙zm╘{m╒|m╓}m╫~m╪m┘	Аm┌	Бm█	Вm▄	Гm▌	Дm▐	Еm▀	Жmр	Зmс	Иmт	Йmу	Кmф	Лmх	Мmц	Нmч	Оmш	Пmщ	Рmъ	Сmы	Тmь	Уmэ	ФmюPvяQvЁ^vё`vЄbvєdvЇjvїkvЎuvўvv°wv∙xv·yv√zv№{v¤|v■}v ~vАvБ	АvВ	БvГ	ВvД	ГvЕ	ДvЖ	ЕvЗ	ЖvИ	ЗvЙ	ИvК	ЙvЛ	КvМ	ЛvН	МvО	НvП	ОvР	ПvС	РvТ	СvУ	ТvФ	УvХ	ФvЦ
 
█
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
P26
Q27
R28
S29
П30
Р31
С32
Т33
У34
Ф35
^36
`37
b38
d39
j40
k41
╦
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
P26
Q27
П28
Р29
С30
Т31
У32
Ф33
^34
`35
b36
d37
j38
k39
▓
Хlayers
Цnon_trainable_variables
Чlayer_metrics
	regularization_losses
Шmetrics

	variables
 Щlayer_regularization_losses
trainable_variables
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
u
embeddings
Ъregularization_losses
Ы	variables
Ьtrainable_variables
Э	keras_api
f
v
embeddings
Юregularization_losses
Я	variables
аtrainable_variables
б	keras_api
g
Б
embeddings
вregularization_losses
г	variables
дtrainable_variables
е	keras_api
g
И
embeddings
жregularization_losses
з	variables
иtrainable_variables
й	keras_api
g
Й
embeddings
кregularization_losses
л	variables
мtrainable_variables
н	keras_api
g
К
embeddings
оregularization_losses
п	variables
░trainable_variables
▒	keras_api
g
Л
embeddings
▓regularization_losses
│	variables
┤trainable_variables
╡	keras_api
g
М
embeddings
╢regularization_losses
╖	variables
╕trainable_variables
╣	keras_api
g
Н
embeddings
║regularization_losses
╗	variables
╝trainable_variables
╜	keras_api
g
О
embeddings
╛regularization_losses
┐	variables
└trainable_variables
┴	keras_api
f
w
embeddings
┬regularization_losses
├	variables
─trainable_variables
┼	keras_api
f
x
embeddings
╞regularization_losses
╟	variables
╚trainable_variables
╔	keras_api
f
y
embeddings
╩regularization_losses
╦	variables
╠trainable_variables
═	keras_api
f
z
embeddings
╬regularization_losses
╧	variables
╨trainable_variables
╤	keras_api
f
{
embeddings
╥regularization_losses
╙	variables
╘trainable_variables
╒	keras_api
f
|
embeddings
╓regularization_losses
╫	variables
╪trainable_variables
┘	keras_api
f
}
embeddings
┌regularization_losses
█	variables
▄trainable_variables
▌	keras_api
f
~
embeddings
▐regularization_losses
▀	variables
рtrainable_variables
с	keras_api
f

embeddings
тregularization_losses
у	variables
фtrainable_variables
х	keras_api
g
А
embeddings
цregularization_losses
ч	variables
шtrainable_variables
щ	keras_api
g
В
embeddings
ъregularization_losses
ы	variables
ьtrainable_variables
э	keras_api
g
Г
embeddings
юregularization_losses
я	variables
Ёtrainable_variables
ё	keras_api
g
Д
embeddings
Єregularization_losses
є	variables
Їtrainable_variables
ї	keras_api
g
Е
embeddings
Ўregularization_losses
ў	variables
°trainable_variables
∙	keras_api
g
Ж
embeddings
·regularization_losses
√	variables
№trainable_variables
¤	keras_api
g
З
embeddings
■regularization_losses
 	variables
Аtrainable_variables
Б	keras_api
 
sq
VARIABLE_VALUE:dcn__attention__parallel__v1_1/batch_normalization_1/gamma#bn/gamma/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE9dcn__attention__parallel__v1_1/batch_normalization_1/beta"bn/beta/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE@dcn__attention__parallel__v1_1/batch_normalization_1/moving_mean)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUEDdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1
R2
S3

P0
Q1
▓
Вlayers
Гnon_trainable_variables
Дlayer_metrics
Tregularization_losses
Еmetrics
U	variables
 Жlayer_regularization_losses
Vtrainable_variables

З0
И1
n
Уkernel
	Фbias
Йregularization_losses
К	variables
Лtrainable_variables
М	keras_api
 
0
П0
Р1
С2
Т3
У4
Ф5
0
П0
Р1
С2
Т3
У4
Ф5
▓
Нlayers
Оnon_trainable_variables
Пlayer_metrics
Zregularization_losses
Рmetrics
[	variables
 Сlayer_regularization_losses
\trainable_variables
pn
VARIABLE_VALUE0dcn__attention__parallel__v1_1/cross_layer_1/wq0*cross_layer/wq0/.ATTRIBUTES/VARIABLE_VALUE

^0
pn
VARIABLE_VALUE0dcn__attention__parallel__v1_1/cross_layer_1/wk0*cross_layer/wk0/.ATTRIBUTES/VARIABLE_VALUE

`0
pn
VARIABLE_VALUE0dcn__attention__parallel__v1_1/cross_layer_1/wv0*cross_layer/wv0/.ATTRIBUTES/VARIABLE_VALUE

b0
nl
VARIABLE_VALUE/dcn__attention__parallel__v1_1/cross_layer_1/b0)cross_layer/b0/.ATTRIBUTES/VARIABLE_VALUE

d0
 

^0
`1
b2
d3

^0
`1
b2
d3
▓
Тlayers
Уnon_trainable_variables
Фlayer_metrics
fregularization_losses
Хmetrics
g	variables
 Цlayer_regularization_losses
htrainable_variables
qo
VARIABLE_VALUE-dcn__attention__parallel__v1_1/dense_7/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE+dcn__attention__parallel__v1_1/dense_7/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
▓
Чlayers
Шnon_trainable_variables
Щlayer_metrics
lregularization_losses
Ъmetrics
m	variables
 Ыlayer_regularization_losses
ntrainable_variables
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
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_26/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_27/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_36/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_37/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_38/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_39/embeddings&variables/5/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_40/embeddings&variables/6/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_41/embeddings&variables/7/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_42/embeddings&variables/8/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_43/embeddings&variables/9/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_44/embeddings'variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_45/embeddings'variables/11/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_28/embeddings'variables/12/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_46/embeddings'variables/13/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_47/embeddings'variables/14/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_48/embeddings'variables/15/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_49/embeddings'variables/16/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_50/embeddings'variables/17/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_51/embeddings'variables/18/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_29/embeddings'variables/19/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_30/embeddings'variables/20/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_31/embeddings'variables/21/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_32/embeddings'variables/22/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_33/embeddings'variables/23/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_34/embeddings'variables/24/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE6dcn__attention__parallel__v1_1/embedding_35/embeddings'variables/25/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel'variables/34/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias'variables/35/.ATTRIBUTES/VARIABLE_VALUE
ц
50
61
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
712
I13
J14
K15
L16
M17
N18
819
920
:21
;22
<23
=24
>25
26
27
28
29

R0
S1
 

Ь0
Э1
Ю2
 
 

u0

u0
╡
Яlayers
аnon_trainable_variables
бlayer_metrics
Ъregularization_losses
вmetrics
Ы	variables
 гlayer_regularization_losses
Ьtrainable_variables
 

v0

v0
╡
дlayers
еnon_trainable_variables
жlayer_metrics
Юregularization_losses
зmetrics
Я	variables
 иlayer_regularization_losses
аtrainable_variables
 

Б0

Б0
╡
йlayers
кnon_trainable_variables
лlayer_metrics
вregularization_losses
мmetrics
г	variables
 нlayer_regularization_losses
дtrainable_variables
 

И0

И0
╡
оlayers
пnon_trainable_variables
░layer_metrics
жregularization_losses
▒metrics
з	variables
 ▓layer_regularization_losses
иtrainable_variables
 

Й0

Й0
╡
│layers
┤non_trainable_variables
╡layer_metrics
кregularization_losses
╢metrics
л	variables
 ╖layer_regularization_losses
мtrainable_variables
 

К0

К0
╡
╕layers
╣non_trainable_variables
║layer_metrics
оregularization_losses
╗metrics
п	variables
 ╝layer_regularization_losses
░trainable_variables
 

Л0

Л0
╡
╜layers
╛non_trainable_variables
┐layer_metrics
▓regularization_losses
└metrics
│	variables
 ┴layer_regularization_losses
┤trainable_variables
 

М0

М0
╡
┬layers
├non_trainable_variables
─layer_metrics
╢regularization_losses
┼metrics
╖	variables
 ╞layer_regularization_losses
╕trainable_variables
 

Н0

Н0
╡
╟layers
╚non_trainable_variables
╔layer_metrics
║regularization_losses
╩metrics
╗	variables
 ╦layer_regularization_losses
╝trainable_variables
 

О0

О0
╡
╠layers
═non_trainable_variables
╬layer_metrics
╛regularization_losses
╧metrics
┐	variables
 ╨layer_regularization_losses
└trainable_variables
 

w0

w0
╡
╤layers
╥non_trainable_variables
╙layer_metrics
┬regularization_losses
╘metrics
├	variables
 ╒layer_regularization_losses
─trainable_variables
 

x0

x0
╡
╓layers
╫non_trainable_variables
╪layer_metrics
╞regularization_losses
┘metrics
╟	variables
 ┌layer_regularization_losses
╚trainable_variables
 

y0

y0
╡
█layers
▄non_trainable_variables
▌layer_metrics
╩regularization_losses
▐metrics
╦	variables
 ▀layer_regularization_losses
╠trainable_variables
 

z0

z0
╡
рlayers
сnon_trainable_variables
тlayer_metrics
╬regularization_losses
уmetrics
╧	variables
 фlayer_regularization_losses
╨trainable_variables
 

{0

{0
╡
хlayers
цnon_trainable_variables
чlayer_metrics
╥regularization_losses
шmetrics
╙	variables
 щlayer_regularization_losses
╘trainable_variables
 

|0

|0
╡
ъlayers
ыnon_trainable_variables
ьlayer_metrics
╓regularization_losses
эmetrics
╫	variables
 юlayer_regularization_losses
╪trainable_variables
 

}0

}0
╡
яlayers
Ёnon_trainable_variables
ёlayer_metrics
┌regularization_losses
Єmetrics
█	variables
 єlayer_regularization_losses
▄trainable_variables
 

~0

~0
╡
Їlayers
їnon_trainable_variables
Ўlayer_metrics
▐regularization_losses
ўmetrics
▀	variables
 °layer_regularization_losses
рtrainable_variables
 

0

0
╡
∙layers
·non_trainable_variables
√layer_metrics
тregularization_losses
№metrics
у	variables
 ¤layer_regularization_losses
фtrainable_variables
 

А0

А0
╡
■layers
 non_trainable_variables
Аlayer_metrics
цregularization_losses
Бmetrics
ч	variables
 Вlayer_regularization_losses
шtrainable_variables
 

В0

В0
╡
Гlayers
Дnon_trainable_variables
Еlayer_metrics
ъregularization_losses
Жmetrics
ы	variables
 Зlayer_regularization_losses
ьtrainable_variables
 

Г0

Г0
╡
Иlayers
Йnon_trainable_variables
Кlayer_metrics
юregularization_losses
Лmetrics
я	variables
 Мlayer_regularization_losses
Ёtrainable_variables
 

Д0

Д0
╡
Нlayers
Оnon_trainable_variables
Пlayer_metrics
Єregularization_losses
Рmetrics
є	variables
 Сlayer_regularization_losses
Їtrainable_variables
 

Е0

Е0
╡
Тlayers
Уnon_trainable_variables
Фlayer_metrics
Ўregularization_losses
Хmetrics
ў	variables
 Цlayer_regularization_losses
°trainable_variables
 

Ж0

Ж0
╡
Чlayers
Шnon_trainable_variables
Щlayer_metrics
·regularization_losses
Ъmetrics
√	variables
 Ыlayer_regularization_losses
№trainable_variables
 

З0

З0
╡
Ьlayers
Эnon_trainable_variables
Юlayer_metrics
■regularization_losses
Яmetrics
 	variables
 аlayer_regularization_losses
Аtrainable_variables
 

R0
S1
 
 
 
n
Пkernel
	Рbias
бregularization_losses
в	variables
гtrainable_variables
д	keras_api
n
Сkernel
	Тbias
еregularization_losses
ж	variables
зtrainable_variables
и	keras_api
 

У0
Ф1

У0
Ф1
╡
йlayers
кnon_trainable_variables
лlayer_metrics
Йregularization_losses
мmetrics
К	variables
 нlayer_regularization_losses
Лtrainable_variables

З0
И1
Y2
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
8

оtotal

пcount
░	variables
▒	keras_api
v
▓true_positives
│true_negatives
┤false_positives
╡false_negatives
╢	variables
╖	keras_api
\
╕
thresholds
╣true_positives
║false_negatives
╗	variables
╝	keras_api
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
П0
Р1

П0
Р1
╡
╜layers
╛non_trainable_variables
┐layer_metrics
бregularization_losses
└metrics
в	variables
 ┴layer_regularization_losses
гtrainable_variables
 

С0
Т1

С0
Т1
╡
┬layers
├non_trainable_variables
─layer_metrics
еregularization_losses
┼metrics
ж	variables
 ╞layer_regularization_losses
зtrainable_variables
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
о0
п1

░	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
▓0
│1
┤2
╡3

╢	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

╣0
║1

╗	variables
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
ЧФ
VARIABLE_VALUEAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/mFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/mFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/mFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/mEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/mJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE2Adam/dcn__attention__parallel__v1_1/dense_7/bias/mHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/mCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/mCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUEAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/vFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/vFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ФС
VARIABLE_VALUE7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/vFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ТП
VARIABLE_VALUE6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/vEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/vJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE2Adam/dcn__attention__parallel__v1_1/dense_7/bias/vHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUEBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/vCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/vCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:         '*
dtype0*
shape:         '
┬
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16dcn__attention__parallel__v1_1/embedding_26/embeddings6dcn__attention__parallel__v1_1/embedding_27/embeddings6dcn__attention__parallel__v1_1/embedding_28/embeddings6dcn__attention__parallel__v1_1/embedding_29/embeddings6dcn__attention__parallel__v1_1/embedding_30/embeddings6dcn__attention__parallel__v1_1/embedding_31/embeddings6dcn__attention__parallel__v1_1/embedding_32/embeddings6dcn__attention__parallel__v1_1/embedding_33/embeddings6dcn__attention__parallel__v1_1/embedding_34/embeddings6dcn__attention__parallel__v1_1/embedding_35/embeddings6dcn__attention__parallel__v1_1/embedding_36/embeddings6dcn__attention__parallel__v1_1/embedding_37/embeddings6dcn__attention__parallel__v1_1/embedding_38/embeddings6dcn__attention__parallel__v1_1/embedding_39/embeddings6dcn__attention__parallel__v1_1/embedding_40/embeddings6dcn__attention__parallel__v1_1/embedding_41/embeddings6dcn__attention__parallel__v1_1/embedding_42/embeddings6dcn__attention__parallel__v1_1/embedding_43/embeddings6dcn__attention__parallel__v1_1/embedding_44/embeddings6dcn__attention__parallel__v1_1/embedding_45/embeddings6dcn__attention__parallel__v1_1/embedding_46/embeddings6dcn__attention__parallel__v1_1/embedding_47/embeddings6dcn__attention__parallel__v1_1/embedding_48/embeddings6dcn__attention__parallel__v1_1/embedding_49/embeddings6dcn__attention__parallel__v1_1/embedding_50/embeddings6dcn__attention__parallel__v1_1/embedding_51/embeddings@dcn__attention__parallel__v1_1/batch_normalization_1/moving_meanDdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance9dcn__attention__parallel__v1_1/batch_normalization_1/beta:dcn__attention__parallel__v1_1/batch_normalization_1/gamma0dcn__attention__parallel__v1_1/cross_layer_1/wk00dcn__attention__parallel__v1_1/cross_layer_1/wq00dcn__attention__parallel__v1_1/cross_layer_1/wv0/dcn__attention__parallel__v1_1/cross_layer_1/b0;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias-dcn__attention__parallel__v1_1/dense_7/kernel+dcn__attention__parallel__v1_1/dense_7/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_64404
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╝S
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameNdcn__attention__parallel__v1_1/batch_normalization_1/gamma/Read/ReadVariableOpMdcn__attention__parallel__v1_1/batch_normalization_1/beta/Read/ReadVariableOpTdcn__attention__parallel__v1_1/batch_normalization_1/moving_mean/Read/ReadVariableOpXdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance/Read/ReadVariableOpDdcn__attention__parallel__v1_1/cross_layer_1/wq0/Read/ReadVariableOpDdcn__attention__parallel__v1_1/cross_layer_1/wk0/Read/ReadVariableOpDdcn__attention__parallel__v1_1/cross_layer_1/wv0/Read/ReadVariableOpCdcn__attention__parallel__v1_1/cross_layer_1/b0/Read/ReadVariableOpAdcn__attention__parallel__v1_1/dense_7/kernel/Read/ReadVariableOp?dcn__attention__parallel__v1_1/dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_26/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_27/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_36/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_37/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_38/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_39/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_40/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_41/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_42/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_43/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_44/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_45/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_28/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_46/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_47/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_48/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_49/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_50/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_51/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_29/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_30/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_31/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_32/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_33/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_34/embeddings/Read/ReadVariableOpJdcn__attention__parallel__v1_1/embedding_35/embeddings/Read/ReadVariableOpOdcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/Read/ReadVariableOpMdcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/Read/ReadVariableOpOdcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/Read/ReadVariableOpMdcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/Read/ReadVariableOpOdcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/Read/ReadVariableOpMdcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOpUAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m/Read/ReadVariableOpJAdam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m/Read/ReadVariableOpHAdam/dcn__attention__parallel__v1_1/dense_7/kernel/m/Read/ReadVariableOpFAdam/dcn__attention__parallel__v1_1/dense_7/bias/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_35/embeddings/m/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/m/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/m/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/m/Read/ReadVariableOpUAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v/Read/ReadVariableOpKAdam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v/Read/ReadVariableOpJAdam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v/Read/ReadVariableOpHAdam/dcn__attention__parallel__v1_1/dense_7/kernel/v/Read/ReadVariableOpFAdam/dcn__attention__parallel__v1_1/dense_7/bias/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v/Read/ReadVariableOpQAdam/dcn__attention__parallel__v1_1/embedding_35/embeddings/v/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/v/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/v/Read/ReadVariableOpVAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v/Read/ReadVariableOpTAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v/Read/ReadVariableOpConst*Ч
TinП
М2Й	*
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
GPU2*0J 8В *'
f"R 
__inference__traced_save_66579
л>
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename:dcn__attention__parallel__v1_1/batch_normalization_1/gamma9dcn__attention__parallel__v1_1/batch_normalization_1/beta@dcn__attention__parallel__v1_1/batch_normalization_1/moving_meanDdcn__attention__parallel__v1_1/batch_normalization_1/moving_variance0dcn__attention__parallel__v1_1/cross_layer_1/wq00dcn__attention__parallel__v1_1/cross_layer_1/wk00dcn__attention__parallel__v1_1/cross_layer_1/wv0/dcn__attention__parallel__v1_1/cross_layer_1/b0-dcn__attention__parallel__v1_1/dense_7/kernel+dcn__attention__parallel__v1_1/dense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate6dcn__attention__parallel__v1_1/embedding_26/embeddings6dcn__attention__parallel__v1_1/embedding_27/embeddings6dcn__attention__parallel__v1_1/embedding_36/embeddings6dcn__attention__parallel__v1_1/embedding_37/embeddings6dcn__attention__parallel__v1_1/embedding_38/embeddings6dcn__attention__parallel__v1_1/embedding_39/embeddings6dcn__attention__parallel__v1_1/embedding_40/embeddings6dcn__attention__parallel__v1_1/embedding_41/embeddings6dcn__attention__parallel__v1_1/embedding_42/embeddings6dcn__attention__parallel__v1_1/embedding_43/embeddings6dcn__attention__parallel__v1_1/embedding_44/embeddings6dcn__attention__parallel__v1_1/embedding_45/embeddings6dcn__attention__parallel__v1_1/embedding_28/embeddings6dcn__attention__parallel__v1_1/embedding_46/embeddings6dcn__attention__parallel__v1_1/embedding_47/embeddings6dcn__attention__parallel__v1_1/embedding_48/embeddings6dcn__attention__parallel__v1_1/embedding_49/embeddings6dcn__attention__parallel__v1_1/embedding_50/embeddings6dcn__attention__parallel__v1_1/embedding_51/embeddings6dcn__attention__parallel__v1_1/embedding_29/embeddings6dcn__attention__parallel__v1_1/embedding_30/embeddings6dcn__attention__parallel__v1_1/embedding_31/embeddings6dcn__attention__parallel__v1_1/embedding_32/embeddings6dcn__attention__parallel__v1_1/embedding_33/embeddings6dcn__attention__parallel__v1_1/embedding_34/embeddings6dcn__attention__parallel__v1_1/embedding_35/embeddings;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel9dcn__attention__parallel__v1_1/dense_layer_1/dense_6/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1false_negatives_1AAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/m2Adam/dcn__attention__parallel__v1_1/dense_7/bias/m=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/mBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/mBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/mBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/mAAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v7Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v6Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v4Adam/dcn__attention__parallel__v1_1/dense_7/kernel/v2Adam/dcn__attention__parallel__v1_1/dense_7/bias/v=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/vBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/vBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/vBAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v*Ц
TinО
Л2И*
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
GPU2*0J 8В **
f%R#
!__inference__traced_restore_66994шн/
╒
є
__inference_loss_fn_0_65676n
[dcn__attention__parallel__v1_1_cross_layer_1_wq0_regularizer_square_readvariableop_resource:	▌
identityИвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp┼
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOp[dcn__attention__parallel__v1_1_cross_layer_1_wq0_regularizer_square_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulО
IdentityIdentityDdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityг
NoOpNoOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp
▐
╘
5__inference_batch_normalization_1_layer_call_fn_65412

inputs
unknown:	▌
	unknown_0:	▌
	unknown_1:	▌
	unknown_2:	▌
identityИвStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_26_layer_call_and_return_conditional_losses_65719

inputs(
embedding_lookup_65713:h
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65713Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65713*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65713*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_45_layer_call_and_return_conditional_losses_66042

inputs(
embedding_lookup_66036:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66036Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66036*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66036*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_49_layer_call_and_return_conditional_losses_62595

inputs)
embedding_lookup_62589:	╕
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62589Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62589*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62589*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_43_layer_call_and_return_conditional_losses_62487

inputs)
embedding_lookup_62481:	Г
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62481Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62481*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62481*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_36_layer_call_fn_65896

inputs
unknown:	Т
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_623612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_28_layer_call_and_return_conditional_losses_62217

inputs)
embedding_lookup_62211:	╬
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62211Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62211*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62211*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_34_layer_call_and_return_conditional_losses_65855

inputs(
embedding_lookup_65849:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65849Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65849*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65849*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_40_layer_call_and_return_conditional_losses_62433

inputs)
embedding_lookup_62427:	╪

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62427Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62427*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62427*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
з	
З
-__inference_dense_layer_1_layer_call_fn_65494

inputs
unknown:	▌ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_629322
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
Ё
■
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65460

inputs9
&dense_4_matmul_readvariableop_resource:	▌ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02
dense_4/MatMul/ReadVariableOpЛ
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/BiasAdds
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityС
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_51_layer_call_fn_66151

inputs
unknown:	╩
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_626312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_36_layer_call_and_return_conditional_losses_62361

inputs)
embedding_lookup_62355:	Т
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62355Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62355*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62355*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_51_layer_call_and_return_conditional_losses_62631

inputs)
embedding_lookup_62625:	╩
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62625Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62625*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62625*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_34_layer_call_fn_65862

inputs
unknown:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_623252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_27_layer_call_and_return_conditional_losses_62199

inputs)
embedding_lookup_62193:	 
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62193Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62193*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62193*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_39_layer_call_and_return_conditional_losses_62415

inputs(
embedding_lookup_62409:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62409Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62409*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62409*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_44_layer_call_and_return_conditional_losses_66025

inputs)
embedding_lookup_66019:	└
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66019Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66019*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66019*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_41_layer_call_and_return_conditional_losses_65974

inputs)
embedding_lookup_65968:	╔
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65968Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65968*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65968*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_35_layer_call_and_return_conditional_losses_65872

inputs)
embedding_lookup_65866:	╕
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65866Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65866*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65866*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_40_layer_call_and_return_conditional_losses_65957

inputs)
embedding_lookup_65951:	╪

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65951Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65951*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65951*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
Ї 
╬z
!__inference__traced_restore_66994
file_prefixZ
Kassignvariableop_dcn__attention__parallel__v1_1_batch_normalization_1_gamma:	▌[
Lassignvariableop_1_dcn__attention__parallel__v1_1_batch_normalization_1_beta:	▌b
Sassignvariableop_2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_mean:	▌f
Wassignvariableop_3_dcn__attention__parallel__v1_1_batch_normalization_1_moving_variance:	▌V
Cassignvariableop_4_dcn__attention__parallel__v1_1_cross_layer_1_wq0:	▌V
Cassignvariableop_5_dcn__attention__parallel__v1_1_cross_layer_1_wk0:	▌V
Cassignvariableop_6_dcn__attention__parallel__v1_1_cross_layer_1_wv0:	▌U
Bassignvariableop_7_dcn__attention__parallel__v1_1_cross_layer_1_b0:	▌S
@assignvariableop_8_dcn__attention__parallel__v1_1_dense_7_kernel:	▐L
>assignvariableop_9_dcn__attention__parallel__v1_1_dense_7_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: \
Jassignvariableop_15_dcn__attention__parallel__v1_1_embedding_26_embeddings:h]
Jassignvariableop_16_dcn__attention__parallel__v1_1_embedding_27_embeddings:	 ]
Jassignvariableop_17_dcn__attention__parallel__v1_1_embedding_36_embeddings:	Т]
Jassignvariableop_18_dcn__attention__parallel__v1_1_embedding_37_embeddings:	х]
Jassignvariableop_19_dcn__attention__parallel__v1_1_embedding_38_embeddings:	┌\
Jassignvariableop_20_dcn__attention__parallel__v1_1_embedding_39_embeddings:]
Jassignvariableop_21_dcn__attention__parallel__v1_1_embedding_40_embeddings:	╪
]
Jassignvariableop_22_dcn__attention__parallel__v1_1_embedding_41_embeddings:	╔\
Jassignvariableop_23_dcn__attention__parallel__v1_1_embedding_42_embeddings:
]
Jassignvariableop_24_dcn__attention__parallel__v1_1_embedding_43_embeddings:	Г]
Jassignvariableop_25_dcn__attention__parallel__v1_1_embedding_44_embeddings:	└\
Jassignvariableop_26_dcn__attention__parallel__v1_1_embedding_45_embeddings:]
Jassignvariableop_27_dcn__attention__parallel__v1_1_embedding_28_embeddings:	╬]
Jassignvariableop_28_dcn__attention__parallel__v1_1_embedding_46_embeddings:	З\
Jassignvariableop_29_dcn__attention__parallel__v1_1_embedding_47_embeddings:
\
Jassignvariableop_30_dcn__attention__parallel__v1_1_embedding_48_embeddings:]
Jassignvariableop_31_dcn__attention__parallel__v1_1_embedding_49_embeddings:	╕\
Jassignvariableop_32_dcn__attention__parallel__v1_1_embedding_50_embeddings:']
Jassignvariableop_33_dcn__attention__parallel__v1_1_embedding_51_embeddings:	╩]
Jassignvariableop_34_dcn__attention__parallel__v1_1_embedding_29_embeddings:	┼\
Jassignvariableop_35_dcn__attention__parallel__v1_1_embedding_30_embeddings:*\
Jassignvariableop_36_dcn__attention__parallel__v1_1_embedding_31_embeddings:	]
Jassignvariableop_37_dcn__attention__parallel__v1_1_embedding_32_embeddings:	╨\
Jassignvariableop_38_dcn__attention__parallel__v1_1_embedding_33_embeddings:C\
Jassignvariableop_39_dcn__attention__parallel__v1_1_embedding_34_embeddings:]
Jassignvariableop_40_dcn__attention__parallel__v1_1_embedding_35_embeddings:	╕b
Oassignvariableop_41_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel:	▌ [
Massignvariableop_42_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias: a
Oassignvariableop_43_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel:  [
Massignvariableop_44_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias: a
Oassignvariableop_45_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel: [
Massignvariableop_46_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias:#
assignvariableop_47_total: #
assignvariableop_48_count: 1
"assignvariableop_49_true_positives:	╚1
"assignvariableop_50_true_negatives:	╚2
#assignvariableop_51_false_positives:	╚2
#assignvariableop_52_false_negatives:	╚2
$assignvariableop_53_true_positives_1:3
%assignvariableop_54_false_negatives_1:d
Uassignvariableop_55_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_m:	▌c
Tassignvariableop_56_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_m:	▌^
Kassignvariableop_57_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_m:	▌^
Kassignvariableop_58_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_m:	▌^
Kassignvariableop_59_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_m:	▌]
Jassignvariableop_60_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_m:	▌[
Hassignvariableop_61_adam_dcn__attention__parallel__v1_1_dense_7_kernel_m:	▐T
Fassignvariableop_62_adam_dcn__attention__parallel__v1_1_dense_7_bias_m:c
Qassignvariableop_63_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_m:hd
Qassignvariableop_64_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_m:	 d
Qassignvariableop_65_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_m:	Тd
Qassignvariableop_66_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_m:	хd
Qassignvariableop_67_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_m:	┌c
Qassignvariableop_68_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_m:d
Qassignvariableop_69_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_m:	╪
d
Qassignvariableop_70_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_m:	╔c
Qassignvariableop_71_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_m:
d
Qassignvariableop_72_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_m:	Гd
Qassignvariableop_73_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_m:	└c
Qassignvariableop_74_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_m:d
Qassignvariableop_75_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_m:	╬d
Qassignvariableop_76_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_m:	Зc
Qassignvariableop_77_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_m:
c
Qassignvariableop_78_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_m:d
Qassignvariableop_79_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_m:	╕c
Qassignvariableop_80_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_m:'d
Qassignvariableop_81_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_m:	╩d
Qassignvariableop_82_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_m:	┼c
Qassignvariableop_83_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_m:*c
Qassignvariableop_84_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_m:	d
Qassignvariableop_85_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_m:	╨c
Qassignvariableop_86_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_m:Cc
Qassignvariableop_87_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_m:d
Qassignvariableop_88_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_m:	╕i
Vassignvariableop_89_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_m:	▌ b
Tassignvariableop_90_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_m: h
Vassignvariableop_91_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_m:  b
Tassignvariableop_92_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_m: h
Vassignvariableop_93_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_m: b
Tassignvariableop_94_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_m:d
Uassignvariableop_95_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_v:	▌c
Tassignvariableop_96_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_v:	▌^
Kassignvariableop_97_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_v:	▌^
Kassignvariableop_98_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_v:	▌^
Kassignvariableop_99_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_v:	▌^
Kassignvariableop_100_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_v:	▌\
Iassignvariableop_101_adam_dcn__attention__parallel__v1_1_dense_7_kernel_v:	▐U
Gassignvariableop_102_adam_dcn__attention__parallel__v1_1_dense_7_bias_v:d
Rassignvariableop_103_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_v:he
Rassignvariableop_104_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_v:	 e
Rassignvariableop_105_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_v:	Тe
Rassignvariableop_106_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_v:	хe
Rassignvariableop_107_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_v:	┌d
Rassignvariableop_108_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_v:e
Rassignvariableop_109_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_v:	╪
e
Rassignvariableop_110_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_v:	╔d
Rassignvariableop_111_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_v:
e
Rassignvariableop_112_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_v:	Гe
Rassignvariableop_113_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_v:	└d
Rassignvariableop_114_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_v:e
Rassignvariableop_115_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_v:	╬e
Rassignvariableop_116_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_v:	Зd
Rassignvariableop_117_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_v:
d
Rassignvariableop_118_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_v:e
Rassignvariableop_119_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_v:	╕d
Rassignvariableop_120_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_v:'e
Rassignvariableop_121_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_v:	╩e
Rassignvariableop_122_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_v:	┼d
Rassignvariableop_123_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_v:*d
Rassignvariableop_124_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_v:	e
Rassignvariableop_125_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_v:	╨d
Rassignvariableop_126_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_v:Cd
Rassignvariableop_127_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_v:e
Rassignvariableop_128_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_v:	╕j
Wassignvariableop_129_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_v:	▌ c
Uassignvariableop_130_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_v: i
Wassignvariableop_131_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_v:  c
Uassignvariableop_132_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_v: i
Wassignvariableop_133_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_v: c
Uassignvariableop_134_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_v:
identity_136ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_126вAssignVariableOp_127вAssignVariableOp_128вAssignVariableOp_129вAssignVariableOp_13вAssignVariableOp_130вAssignVariableOp_131вAssignVariableOp_132вAssignVariableOp_133вAssignVariableOp_134вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99╒?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:И*
dtype0*р>
value╓>B╙>ИB#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wq0/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wk0/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wv0/.ATTRIBUTES/VARIABLE_VALUEB)cross_layer/b0/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesг
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:И*
dtype0*ж
valueЬBЩИB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╢
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Щ
dtypesО
Л2И	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity╩
AssignVariableOpAssignVariableOpKassignvariableop_dcn__attention__parallel__v1_1_batch_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1╤
AssignVariableOp_1AssignVariableOpLassignvariableop_1_dcn__attention__parallel__v1_1_batch_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2╪
AssignVariableOp_2AssignVariableOpSassignvariableop_2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3▄
AssignVariableOp_3AssignVariableOpWassignvariableop_3_dcn__attention__parallel__v1_1_batch_normalization_1_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╚
AssignVariableOp_4AssignVariableOpCassignvariableop_4_dcn__attention__parallel__v1_1_cross_layer_1_wq0Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╚
AssignVariableOp_5AssignVariableOpCassignvariableop_5_dcn__attention__parallel__v1_1_cross_layer_1_wk0Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╚
AssignVariableOp_6AssignVariableOpCassignvariableop_6_dcn__attention__parallel__v1_1_cross_layer_1_wv0Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╟
AssignVariableOp_7AssignVariableOpBassignvariableop_7_dcn__attention__parallel__v1_1_cross_layer_1_b0Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8┼
AssignVariableOp_8AssignVariableOp@assignvariableop_8_dcn__attention__parallel__v1_1_dense_7_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9├
AssignVariableOp_9AssignVariableOp>assignvariableop_9_dcn__attention__parallel__v1_1_dense_7_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10е
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11з
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12з
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ж
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14о
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╥
AssignVariableOp_15AssignVariableOpJassignvariableop_15_dcn__attention__parallel__v1_1_embedding_26_embeddingsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╥
AssignVariableOp_16AssignVariableOpJassignvariableop_16_dcn__attention__parallel__v1_1_embedding_27_embeddingsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╥
AssignVariableOp_17AssignVariableOpJassignvariableop_17_dcn__attention__parallel__v1_1_embedding_36_embeddingsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╥
AssignVariableOp_18AssignVariableOpJassignvariableop_18_dcn__attention__parallel__v1_1_embedding_37_embeddingsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╥
AssignVariableOp_19AssignVariableOpJassignvariableop_19_dcn__attention__parallel__v1_1_embedding_38_embeddingsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╥
AssignVariableOp_20AssignVariableOpJassignvariableop_20_dcn__attention__parallel__v1_1_embedding_39_embeddingsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╥
AssignVariableOp_21AssignVariableOpJassignvariableop_21_dcn__attention__parallel__v1_1_embedding_40_embeddingsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╥
AssignVariableOp_22AssignVariableOpJassignvariableop_22_dcn__attention__parallel__v1_1_embedding_41_embeddingsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╥
AssignVariableOp_23AssignVariableOpJassignvariableop_23_dcn__attention__parallel__v1_1_embedding_42_embeddingsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╥
AssignVariableOp_24AssignVariableOpJassignvariableop_24_dcn__attention__parallel__v1_1_embedding_43_embeddingsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╥
AssignVariableOp_25AssignVariableOpJassignvariableop_25_dcn__attention__parallel__v1_1_embedding_44_embeddingsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╥
AssignVariableOp_26AssignVariableOpJassignvariableop_26_dcn__attention__parallel__v1_1_embedding_45_embeddingsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╥
AssignVariableOp_27AssignVariableOpJassignvariableop_27_dcn__attention__parallel__v1_1_embedding_28_embeddingsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╥
AssignVariableOp_28AssignVariableOpJassignvariableop_28_dcn__attention__parallel__v1_1_embedding_46_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╥
AssignVariableOp_29AssignVariableOpJassignvariableop_29_dcn__attention__parallel__v1_1_embedding_47_embeddingsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╥
AssignVariableOp_30AssignVariableOpJassignvariableop_30_dcn__attention__parallel__v1_1_embedding_48_embeddingsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╥
AssignVariableOp_31AssignVariableOpJassignvariableop_31_dcn__attention__parallel__v1_1_embedding_49_embeddingsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╥
AssignVariableOp_32AssignVariableOpJassignvariableop_32_dcn__attention__parallel__v1_1_embedding_50_embeddingsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╥
AssignVariableOp_33AssignVariableOpJassignvariableop_33_dcn__attention__parallel__v1_1_embedding_51_embeddingsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╥
AssignVariableOp_34AssignVariableOpJassignvariableop_34_dcn__attention__parallel__v1_1_embedding_29_embeddingsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╥
AssignVariableOp_35AssignVariableOpJassignvariableop_35_dcn__attention__parallel__v1_1_embedding_30_embeddingsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36╥
AssignVariableOp_36AssignVariableOpJassignvariableop_36_dcn__attention__parallel__v1_1_embedding_31_embeddingsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╥
AssignVariableOp_37AssignVariableOpJassignvariableop_37_dcn__attention__parallel__v1_1_embedding_32_embeddingsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╥
AssignVariableOp_38AssignVariableOpJassignvariableop_38_dcn__attention__parallel__v1_1_embedding_33_embeddingsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╥
AssignVariableOp_39AssignVariableOpJassignvariableop_39_dcn__attention__parallel__v1_1_embedding_34_embeddingsIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╥
AssignVariableOp_40AssignVariableOpJassignvariableop_40_dcn__attention__parallel__v1_1_embedding_35_embeddingsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╫
AssignVariableOp_41AssignVariableOpOassignvariableop_41_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╒
AssignVariableOp_42AssignVariableOpMassignvariableop_42_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╫
AssignVariableOp_43AssignVariableOpOassignvariableop_43_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╒
AssignVariableOp_44AssignVariableOpMassignvariableop_44_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╫
AssignVariableOp_45AssignVariableOpOassignvariableop_45_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╒
AssignVariableOp_46AssignVariableOpMassignvariableop_46_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47б
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48б
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49к
AssignVariableOp_49AssignVariableOp"assignvariableop_49_true_positivesIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50к
AssignVariableOp_50AssignVariableOp"assignvariableop_50_true_negativesIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51л
AssignVariableOp_51AssignVariableOp#assignvariableop_51_false_positivesIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52л
AssignVariableOp_52AssignVariableOp#assignvariableop_52_false_negativesIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53м
AssignVariableOp_53AssignVariableOp$assignvariableop_53_true_positives_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54н
AssignVariableOp_54AssignVariableOp%assignvariableop_54_false_negatives_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55▌
AssignVariableOp_55AssignVariableOpUassignvariableop_55_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56▄
AssignVariableOp_56AssignVariableOpTassignvariableop_56_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57╙
AssignVariableOp_57AssignVariableOpKassignvariableop_57_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58╙
AssignVariableOp_58AssignVariableOpKassignvariableop_58_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59╙
AssignVariableOp_59AssignVariableOpKassignvariableop_59_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60╥
AssignVariableOp_60AssignVariableOpJassignvariableop_60_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61╨
AssignVariableOp_61AssignVariableOpHassignvariableop_61_adam_dcn__attention__parallel__v1_1_dense_7_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62╬
AssignVariableOp_62AssignVariableOpFassignvariableop_62_adam_dcn__attention__parallel__v1_1_dense_7_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63┘
AssignVariableOp_63AssignVariableOpQassignvariableop_63_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64┘
AssignVariableOp_64AssignVariableOpQassignvariableop_64_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65┘
AssignVariableOp_65AssignVariableOpQassignvariableop_65_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66┘
AssignVariableOp_66AssignVariableOpQassignvariableop_66_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67┘
AssignVariableOp_67AssignVariableOpQassignvariableop_67_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68┘
AssignVariableOp_68AssignVariableOpQassignvariableop_68_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69┘
AssignVariableOp_69AssignVariableOpQassignvariableop_69_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70┘
AssignVariableOp_70AssignVariableOpQassignvariableop_70_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71┘
AssignVariableOp_71AssignVariableOpQassignvariableop_71_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72┘
AssignVariableOp_72AssignVariableOpQassignvariableop_72_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73┘
AssignVariableOp_73AssignVariableOpQassignvariableop_73_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74┘
AssignVariableOp_74AssignVariableOpQassignvariableop_74_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75┘
AssignVariableOp_75AssignVariableOpQassignvariableop_75_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76┘
AssignVariableOp_76AssignVariableOpQassignvariableop_76_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77┘
AssignVariableOp_77AssignVariableOpQassignvariableop_77_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78┘
AssignVariableOp_78AssignVariableOpQassignvariableop_78_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79┘
AssignVariableOp_79AssignVariableOpQassignvariableop_79_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80┘
AssignVariableOp_80AssignVariableOpQassignvariableop_80_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81┘
AssignVariableOp_81AssignVariableOpQassignvariableop_81_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82┘
AssignVariableOp_82AssignVariableOpQassignvariableop_82_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83┘
AssignVariableOp_83AssignVariableOpQassignvariableop_83_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84┘
AssignVariableOp_84AssignVariableOpQassignvariableop_84_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85┘
AssignVariableOp_85AssignVariableOpQassignvariableop_85_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86┘
AssignVariableOp_86AssignVariableOpQassignvariableop_86_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87┘
AssignVariableOp_87AssignVariableOpQassignvariableop_87_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88┘
AssignVariableOp_88AssignVariableOpQassignvariableop_88_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89▐
AssignVariableOp_89AssignVariableOpVassignvariableop_89_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90▄
AssignVariableOp_90AssignVariableOpTassignvariableop_90_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91▐
AssignVariableOp_91AssignVariableOpVassignvariableop_91_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92▄
AssignVariableOp_92AssignVariableOpTassignvariableop_92_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93▐
AssignVariableOp_93AssignVariableOpVassignvariableop_93_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94▄
AssignVariableOp_94AssignVariableOpTassignvariableop_94_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95▌
AssignVariableOp_95AssignVariableOpUassignvariableop_95_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96▄
AssignVariableOp_96AssignVariableOpTassignvariableop_96_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97╙
AssignVariableOp_97AssignVariableOpKassignvariableop_97_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98╙
AssignVariableOp_98AssignVariableOpKassignvariableop_98_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99╙
AssignVariableOp_99AssignVariableOpKassignvariableop_99_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100╓
AssignVariableOp_100AssignVariableOpKassignvariableop_100_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101╘
AssignVariableOp_101AssignVariableOpIassignvariableop_101_adam_dcn__attention__parallel__v1_1_dense_7_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102╥
AssignVariableOp_102AssignVariableOpGassignvariableop_102_adam_dcn__attention__parallel__v1_1_dense_7_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103▌
AssignVariableOp_103AssignVariableOpRassignvariableop_103_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104▌
AssignVariableOp_104AssignVariableOpRassignvariableop_104_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105▌
AssignVariableOp_105AssignVariableOpRassignvariableop_105_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106▌
AssignVariableOp_106AssignVariableOpRassignvariableop_106_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107▌
AssignVariableOp_107AssignVariableOpRassignvariableop_107_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108▌
AssignVariableOp_108AssignVariableOpRassignvariableop_108_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109▌
AssignVariableOp_109AssignVariableOpRassignvariableop_109_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110▌
AssignVariableOp_110AssignVariableOpRassignvariableop_110_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111▌
AssignVariableOp_111AssignVariableOpRassignvariableop_111_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112▌
AssignVariableOp_112AssignVariableOpRassignvariableop_112_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113▌
AssignVariableOp_113AssignVariableOpRassignvariableop_113_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114▌
AssignVariableOp_114AssignVariableOpRassignvariableop_114_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115▌
AssignVariableOp_115AssignVariableOpRassignvariableop_115_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116▌
AssignVariableOp_116AssignVariableOpRassignvariableop_116_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117▌
AssignVariableOp_117AssignVariableOpRassignvariableop_117_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118▌
AssignVariableOp_118AssignVariableOpRassignvariableop_118_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119▌
AssignVariableOp_119AssignVariableOpRassignvariableop_119_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120▌
AssignVariableOp_120AssignVariableOpRassignvariableop_120_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121▌
AssignVariableOp_121AssignVariableOpRassignvariableop_121_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122▌
AssignVariableOp_122AssignVariableOpRassignvariableop_122_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123▌
AssignVariableOp_123AssignVariableOpRassignvariableop_123_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124▌
AssignVariableOp_124AssignVariableOpRassignvariableop_124_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125▌
AssignVariableOp_125AssignVariableOpRassignvariableop_125_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126▌
AssignVariableOp_126AssignVariableOpRassignvariableop_126_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127▌
AssignVariableOp_127AssignVariableOpRassignvariableop_127_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128▌
AssignVariableOp_128AssignVariableOpRassignvariableop_128_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129т
AssignVariableOp_129AssignVariableOpWassignvariableop_129_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130р
AssignVariableOp_130AssignVariableOpUassignvariableop_130_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131т
AssignVariableOp_131AssignVariableOpWassignvariableop_131_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132р
AssignVariableOp_132AssignVariableOpUassignvariableop_132_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133т
AssignVariableOp_133AssignVariableOpWassignvariableop_133_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134р
AssignVariableOp_134AssignVariableOpUassignvariableop_134_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЭ
Identity_135Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_135i
Identity_136IdentityIdentity_135:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_136Г
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_136Identity_136:output:0*е
_input_shapesУ
Р: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
█W
╓
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_63014

inputs.
mul_readvariableop_resource:	▌0
mul_1_readvariableop_resource:	▌0
mul_2_readvariableop_resource:	▌.
add_readvariableop_resource:	▌
identityИвMul/ReadVariableOpвMul_1/ReadVariableOpвMul_2/ReadVariableOpвadd/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2

ExpandDimsЕ
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul/ReadVariableOpy
MulMulExpandDims:output:0Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
MulЛ
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_1/ReadVariableOp
Mul_1MulExpandDims:output:0Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_1Л
Mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_2/ReadVariableOp
Mul_2MulExpandDims:output:0Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_2z
MatMulBatchMatMulV2Mul:z:0	Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
MatMulZ
ExpExpMatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
Expp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesВ
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
Suml
truedivRealDivExp:y:0Sum:output:0*
T0*-
_output_shapes
:         ▌▌2	
truediv~
MatMul_1BatchMatMulV2truediv:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2

MatMul_1j
Mul_3Mul	Mul_2:z:0MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
Mul_3Е
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02
add/ReadVariableOpq
addAddV2	Mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
addl
add_1AddV2add:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
add_1r
SqueezeSqueeze	add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2	
SqueezeЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulЕ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulГ
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mull
IdentityIdentitySqueeze:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identity∙
NoOpNoOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp^Mul_2/ReadVariableOp^add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2,
Mul_2/ReadVariableOpMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╩
А
,__inference_embedding_42_layer_call_fn_65998

inputs
unknown:

identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_624692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_29_layer_call_fn_65777

inputs
unknown:	┼
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_622352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_33_layer_call_and_return_conditional_losses_62307

inputs(
embedding_lookup_62301:C
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62301Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62301*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62301*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_46_layer_call_and_return_conditional_losses_62541

inputs)
embedding_lookup_62535:	З
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62535Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62535*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62535*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_33_layer_call_and_return_conditional_losses_65838

inputs(
embedding_lookup_65832:C
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65832Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65832*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65832*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_38_layer_call_and_return_conditional_losses_65923

inputs)
embedding_lookup_65917:	┌
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65917Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65917*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65917*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_31_layer_call_and_return_conditional_losses_65804

inputs(
embedding_lookup_65798:	
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65798Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65798*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65798*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
│В
╝
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_62791

inputs$
embedding_26_62182:h%
embedding_27_62200:	 %
embedding_28_62218:	╬%
embedding_29_62236:	┼$
embedding_30_62254:*$
embedding_31_62272:	%
embedding_32_62290:	╨$
embedding_33_62308:C$
embedding_34_62326:%
embedding_35_62344:	╕%
embedding_36_62362:	Т%
embedding_37_62380:	х%
embedding_38_62398:	┌$
embedding_39_62416:%
embedding_40_62434:	╪
%
embedding_41_62452:	╔$
embedding_42_62470:
%
embedding_43_62488:	Г%
embedding_44_62506:	└$
embedding_45_62524:%
embedding_46_62542:	З$
embedding_47_62560:
$
embedding_48_62578:%
embedding_49_62596:	╕$
embedding_50_62614:'%
embedding_51_62632:	╩*
batch_normalization_1_62639:	▌*
batch_normalization_1_62641:	▌*
batch_normalization_1_62643:	▌*
batch_normalization_1_62645:	▌&
cross_layer_1_62700:	▌&
cross_layer_1_62702:	▌&
cross_layer_1_62704:	▌&
cross_layer_1_62706:	▌&
dense_layer_1_62734:	▌ !
dense_layer_1_62736: %
dense_layer_1_62738:  !
dense_layer_1_62740: %
dense_layer_1_62742: !
dense_layer_1_62744: 
dense_7_62760:	▐
dense_7_62762:
identityИв-batch_normalization_1/StatefulPartitionedCallв%cross_layer_1/StatefulPartitionedCallвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/StatefulPartitionedCallв%dense_layer_1/StatefulPartitionedCallв$embedding_26/StatefulPartitionedCallв$embedding_27/StatefulPartitionedCallв$embedding_28/StatefulPartitionedCallв$embedding_29/StatefulPartitionedCallв$embedding_30/StatefulPartitionedCallв$embedding_31/StatefulPartitionedCallв$embedding_32/StatefulPartitionedCallв$embedding_33/StatefulPartitionedCallв$embedding_34/StatefulPartitionedCallв$embedding_35/StatefulPartitionedCallв$embedding_36/StatefulPartitionedCallв$embedding_37/StatefulPartitionedCallв$embedding_38/StatefulPartitionedCallв$embedding_39/StatefulPartitionedCallв$embedding_40/StatefulPartitionedCallв$embedding_41/StatefulPartitionedCallв$embedding_42/StatefulPartitionedCallв$embedding_43/StatefulPartitionedCallв$embedding_44/StatefulPartitionedCallв$embedding_45/StatefulPartitionedCallв$embedding_46/StatefulPartitionedCallв$embedding_47/StatefulPartitionedCallв$embedding_48/StatefulPartitionedCallв$embedding_49/StatefulPartitionedCallв$embedding_50/StatefulPartitionedCallв$embedding_51/StatefulPartitionedCall{
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
strided_slice/stack_2ї
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2 
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2д
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_62182*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_621812&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3д
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_62200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_621992&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4д
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_62218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_622172&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5д
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_62236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_622352&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6д
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_62254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_622532&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7д
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_62272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_622712&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8д
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_62290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_622892&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9д
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_62308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_623072&
$embedding_33/StatefulPartitionedCallБ
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10е
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_62326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_623252&
$embedding_34/StatefulPartitionedCallБ
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11е
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_62344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_623432&
$embedding_35/StatefulPartitionedCallБ
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12е
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_62362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_623612&
$embedding_36/StatefulPartitionedCallБ
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13е
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_62380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_623792&
$embedding_37/StatefulPartitionedCallБ
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14е
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_62398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_623972&
$embedding_38/StatefulPartitionedCallБ
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15е
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_62416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_624152&
$embedding_39/StatefulPartitionedCallБ
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16е
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_62434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_624332&
$embedding_40/StatefulPartitionedCallБ
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17е
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_62452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_624512&
$embedding_41/StatefulPartitionedCallБ
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18е
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_62470*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_624692&
$embedding_42/StatefulPartitionedCallБ
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19е
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_62488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_624872&
$embedding_43/StatefulPartitionedCallБ
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20е
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_62506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_625052&
$embedding_44/StatefulPartitionedCallБ
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21е
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_62524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_625232&
$embedding_45/StatefulPartitionedCallБ
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22е
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_62542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_625412&
$embedding_46/StatefulPartitionedCallБ
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23е
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_62560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_625592&
$embedding_47/StatefulPartitionedCallБ
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24е
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_62578*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_625772&
$embedding_48/StatefulPartitionedCallБ
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25е
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_62596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_625952&
$embedding_49/StatefulPartitionedCallБ
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26е
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_62614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_626132&
$embedding_50/StatefulPartitionedCallБ
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27е
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_62632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_626312&
$embedding_51/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┤

concatConcatV2-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1Я
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0batch_normalization_1_62639batch_normalization_1_62641batch_normalization_1_62643batch_normalization_1_62645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620142/
-batch_normalization_1/StatefulPartitionedCallМ
%cross_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0cross_layer_1_62700cross_layer_1_62702cross_layer_1_62704cross_layer_1_62706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_626992'
%cross_layer_1/StatefulPartitionedCall╣
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_layer_1_62734dense_layer_1_62736dense_layer_1_62738dense_layer_1_62740dense_layer_1_62742dense_layer_1_62744*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_627332'
%dense_layer_1/StatefulPartitionedCall`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╘
concat_2ConcatV2.cross_layer_1/StatefulPartitionedCall:output:0.dense_layer_1/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2Ъ
dense_7/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_7_62760dense_7_62762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_627592!
dense_7/StatefulPartitionedCally
SigmoidSigmoid(dense_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2	
Sigmoid¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_62702*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_62700*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_62704*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul√
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_62706*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity╣
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall&^cross_layer_1/StatefulPartitionedCallR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2N
%cross_layer_1/StatefulPartitionedCall%cross_layer_1/StatefulPartitionedCall2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
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
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╠

д
G__inference_embedding_46_layer_call_and_return_conditional_losses_66059

inputs)
embedding_lookup_66053:	З
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66053Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66053*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66053*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
▒В
╝
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_63585

inputs$
embedding_26_63339:h%
embedding_27_63346:	 %
embedding_28_63353:	╬%
embedding_29_63360:	┼$
embedding_30_63367:*$
embedding_31_63374:	%
embedding_32_63381:	╨$
embedding_33_63388:C$
embedding_34_63395:%
embedding_35_63402:	╕%
embedding_36_63409:	Т%
embedding_37_63416:	х%
embedding_38_63423:	┌$
embedding_39_63430:%
embedding_40_63437:	╪
%
embedding_41_63444:	╔$
embedding_42_63451:
%
embedding_43_63458:	Г%
embedding_44_63465:	└$
embedding_45_63472:%
embedding_46_63479:	З$
embedding_47_63486:
$
embedding_48_63493:%
embedding_49_63500:	╕$
embedding_50_63507:'%
embedding_51_63514:	╩*
batch_normalization_1_63521:	▌*
batch_normalization_1_63523:	▌*
batch_normalization_1_63525:	▌*
batch_normalization_1_63527:	▌&
cross_layer_1_63530:	▌&
cross_layer_1_63532:	▌&
cross_layer_1_63534:	▌&
cross_layer_1_63536:	▌&
dense_layer_1_63539:	▌ !
dense_layer_1_63541: %
dense_layer_1_63543:  !
dense_layer_1_63545: %
dense_layer_1_63547: !
dense_layer_1_63549: 
dense_7_63554:	▐
dense_7_63556:
identityИв-batch_normalization_1/StatefulPartitionedCallв%cross_layer_1/StatefulPartitionedCallвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/StatefulPartitionedCallв%dense_layer_1/StatefulPartitionedCallв$embedding_26/StatefulPartitionedCallв$embedding_27/StatefulPartitionedCallв$embedding_28/StatefulPartitionedCallв$embedding_29/StatefulPartitionedCallв$embedding_30/StatefulPartitionedCallв$embedding_31/StatefulPartitionedCallв$embedding_32/StatefulPartitionedCallв$embedding_33/StatefulPartitionedCallв$embedding_34/StatefulPartitionedCallв$embedding_35/StatefulPartitionedCallв$embedding_36/StatefulPartitionedCallв$embedding_37/StatefulPartitionedCallв$embedding_38/StatefulPartitionedCallв$embedding_39/StatefulPartitionedCallв$embedding_40/StatefulPartitionedCallв$embedding_41/StatefulPartitionedCallв$embedding_42/StatefulPartitionedCallв$embedding_43/StatefulPartitionedCallв$embedding_44/StatefulPartitionedCallв$embedding_45/StatefulPartitionedCallв$embedding_46/StatefulPartitionedCallв$embedding_47/StatefulPartitionedCallв$embedding_48/StatefulPartitionedCallв$embedding_49/StatefulPartitionedCallв$embedding_50/StatefulPartitionedCallв$embedding_51/StatefulPartitionedCall{
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
strided_slice/stack_2ї
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2 
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2д
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_63339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_621812&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3д
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_63346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_621992&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4д
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_63353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_622172&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5д
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_63360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_622352&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6д
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_63367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_622532&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7д
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_63374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_622712&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8д
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_63381*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_622892&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9д
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_63388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_623072&
$embedding_33/StatefulPartitionedCallБ
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10е
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_63395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_623252&
$embedding_34/StatefulPartitionedCallБ
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11е
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_63402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_623432&
$embedding_35/StatefulPartitionedCallБ
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12е
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_63409*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_623612&
$embedding_36/StatefulPartitionedCallБ
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13е
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_63416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_623792&
$embedding_37/StatefulPartitionedCallБ
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14е
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_63423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_623972&
$embedding_38/StatefulPartitionedCallБ
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15е
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_63430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_624152&
$embedding_39/StatefulPartitionedCallБ
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16е
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_63437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_624332&
$embedding_40/StatefulPartitionedCallБ
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17е
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_63444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_624512&
$embedding_41/StatefulPartitionedCallБ
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18е
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_63451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_624692&
$embedding_42/StatefulPartitionedCallБ
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19е
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_63458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_624872&
$embedding_43/StatefulPartitionedCallБ
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20е
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_63465*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_625052&
$embedding_44/StatefulPartitionedCallБ
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21е
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_63472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_625232&
$embedding_45/StatefulPartitionedCallБ
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22е
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_63479*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_625412&
$embedding_46/StatefulPartitionedCallБ
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23е
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_63486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_625592&
$embedding_47/StatefulPartitionedCallБ
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24е
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_63493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_625772&
$embedding_48/StatefulPartitionedCallБ
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25е
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_63500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_625952&
$embedding_49/StatefulPartitionedCallБ
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26е
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_63507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_626132&
$embedding_50/StatefulPartitionedCallБ
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27е
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_63514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_626312&
$embedding_51/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┤

concatConcatV2-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1Э
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0batch_normalization_1_63521batch_normalization_1_63523batch_normalization_1_63525batch_normalization_1_63527*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620742/
-batch_normalization_1/StatefulPartitionedCallМ
%cross_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0cross_layer_1_63530cross_layer_1_63532cross_layer_1_63534cross_layer_1_63536*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_630142'
%cross_layer_1/StatefulPartitionedCall╣
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_layer_1_63539dense_layer_1_63541dense_layer_1_63543dense_layer_1_63545dense_layer_1_63547dense_layer_1_63549*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_629322'
%dense_layer_1/StatefulPartitionedCall`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╘
concat_2ConcatV2.cross_layer_1/StatefulPartitionedCall:output:0.dense_layer_1/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2Ъ
dense_7/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_7_63554dense_7_63556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_627592!
dense_7/StatefulPartitionedCally
SigmoidSigmoid(dense_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2	
Sigmoid¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63532*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63530*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63534*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul√
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63536*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity╣
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall&^cross_layer_1/StatefulPartitionedCallR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2N
%cross_layer_1/StatefulPartitionedCall%cross_layer_1/StatefulPartitionedCall2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
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
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╜
ё
__inference_loss_fn_3_65709m
Zdcn__attention__parallel__v1_1_cross_layer_1_b0_regularizer_square_readvariableop_resource:	▌
identityИвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp┬
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpZdcn__attention__parallel__v1_1_cross_layer_1_b0_regularizer_square_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulН
IdentityIdentityCdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityв
NoOpNoOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp
╠

д
G__inference_embedding_41_layer_call_and_return_conditional_losses_62451

inputs)
embedding_lookup_62445:	╔
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62445Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62445*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62445*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_39_layer_call_fn_65947

inputs
unknown:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_624152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_43_layer_call_and_return_conditional_losses_66008

inputs)
embedding_lookup_66002:	Г
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66002Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66002*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66002*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_30_layer_call_and_return_conditional_losses_62253

inputs(
embedding_lookup_62247:*
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62247Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62247*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62247*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_51_layer_call_and_return_conditional_losses_66144

inputs)
embedding_lookup_66138:	╩
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66138Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66138*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66138*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
║
╨

>__inference_dcn__attention__parallel__v1_1_layer_call_fn_63761
input_1
unknown:h
	unknown_0:	 
	unknown_1:	╬
	unknown_2:	┼
	unknown_3:*
	unknown_4:	
	unknown_5:	╨
	unknown_6:C
	unknown_7:
	unknown_8:	╕
	unknown_9:	Т

unknown_10:	х

unknown_11:	┌

unknown_12:

unknown_13:	╪


unknown_14:	╔

unknown_15:


unknown_16:	Г

unknown_17:	└

unknown_18:

unknown_19:	З

unknown_20:


unknown_21:

unknown_22:	╕

unknown_23:'

unknown_24:	╩

unknown_25:	▌

unknown_26:	▌

unknown_27:	▌

unknown_28:	▌

unknown_29:	▌

unknown_30:	▌

unknown_31:	▌

unknown_32:	▌

unknown_33:	▌ 

unknown_34: 

unknown_35:  

unknown_36: 

unknown_37: 

unknown_38:

unknown_39:	▐

unknown_40:
identityИвStatefulPartitionedCall▓
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
:         *J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_635852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╠

д
G__inference_embedding_35_layer_call_and_return_conditional_losses_62343

inputs)
embedding_lookup_62337:	╕
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62337Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62337*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62337*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_44_layer_call_and_return_conditional_losses_62505

inputs)
embedding_lookup_62499:	└
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62499Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62499*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62499*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_47_layer_call_and_return_conditional_losses_62559

inputs(
embedding_lookup_62553:

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62553Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62553*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62553*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
ш
╡

#__inference_signature_wrapper_64404
input_1
unknown:h
	unknown_0:	 
	unknown_1:	╬
	unknown_2:	┼
	unknown_3:*
	unknown_4:	
	unknown_5:	╨
	unknown_6:C
	unknown_7:
	unknown_8:	╕
	unknown_9:	Т

unknown_10:	х

unknown_11:	┌

unknown_12:

unknown_13:	╪


unknown_14:	╔

unknown_15:


unknown_16:	Г

unknown_17:	└

unknown_18:

unknown_19:	З

unknown_20:


unknown_21:

unknown_22:	╕

unknown_23:'

unknown_24:	╩

unknown_25:	▌

unknown_26:	▌

unknown_27:	▌

unknown_28:	▌

unknown_29:	▌

unknown_30:	▌

unknown_31:	▌

unknown_32:	▌

unknown_33:	▌ 

unknown_34: 

unknown_35:  

unknown_36: 

unknown_37: 

unknown_38:

unknown_39:	▐

unknown_40:
identityИвStatefulPartitionedCall√
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
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_619902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╦

г
G__inference_embedding_26_layer_call_and_return_conditional_losses_62181

inputs(
embedding_lookup_62175:h
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62175Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62175*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62175*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╜
З
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62014

inputs+
cast_readvariableop_resource:	▌-
cast_1_readvariableop_resource:	▌-
cast_2_readvariableop_resource:	▌-
cast_3_readvariableop_resource:	▌
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:▌2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityм
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
є
Х
'__inference_dense_7_layer_call_fn_65665

inputs
unknown:	▐
	unknown_0:
identityИвStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_627592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▐: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▐
 
_user_specified_nameinputs
╜
З
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65352

inputs+
cast_readvariableop_resource:	▌-
cast_1_readvariableop_resource:	▌-
cast_2_readvariableop_resource:	▌-
cast_3_readvariableop_resource:	▌
identityИвCast/ReadVariableOpвCast_1/ReadVariableOpвCast_2/ReadVariableOpвCast_3/ReadVariableOpД
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_1/ReadVariableOpК
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_2/ReadVariableOpК
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_3/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЖ
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:▌2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/mul_1
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2
batchnorm/mul_2
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityм
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_41_layer_call_fn_65981

inputs
unknown:	╔
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_624512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
р
▄
-__inference_cross_layer_1_layer_call_fn_65646

inputs
unknown:	▌
	unknown_0:	▌
	unknown_1:	▌
	unknown_2:	▌
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_630142
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╩
А
,__inference_embedding_26_layer_call_fn_65726

inputs
unknown:h
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_621812
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_49_layer_call_and_return_conditional_losses_66110

inputs)
embedding_lookup_66104:	╕
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66104Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66104*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66104*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_42_layer_call_and_return_conditional_losses_65991

inputs(
embedding_lookup_65985:

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65985Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65985*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65985*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
з	
З
-__inference_dense_layer_1_layer_call_fn_65477

inputs
unknown:	▌ 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_627332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╡В
╜
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64283
input_1$
embedding_26_64037:h%
embedding_27_64044:	 %
embedding_28_64051:	╬%
embedding_29_64058:	┼$
embedding_30_64065:*$
embedding_31_64072:	%
embedding_32_64079:	╨$
embedding_33_64086:C$
embedding_34_64093:%
embedding_35_64100:	╕%
embedding_36_64107:	Т%
embedding_37_64114:	х%
embedding_38_64121:	┌$
embedding_39_64128:%
embedding_40_64135:	╪
%
embedding_41_64142:	╔$
embedding_42_64149:
%
embedding_43_64156:	Г%
embedding_44_64163:	└$
embedding_45_64170:%
embedding_46_64177:	З$
embedding_47_64184:
$
embedding_48_64191:%
embedding_49_64198:	╕$
embedding_50_64205:'%
embedding_51_64212:	╩*
batch_normalization_1_64219:	▌*
batch_normalization_1_64221:	▌*
batch_normalization_1_64223:	▌*
batch_normalization_1_64225:	▌&
cross_layer_1_64228:	▌&
cross_layer_1_64230:	▌&
cross_layer_1_64232:	▌&
cross_layer_1_64234:	▌&
dense_layer_1_64237:	▌ !
dense_layer_1_64239: %
dense_layer_1_64241:  !
dense_layer_1_64243: %
dense_layer_1_64245: !
dense_layer_1_64247: 
dense_7_64252:	▐
dense_7_64254:
identityИв-batch_normalization_1/StatefulPartitionedCallв%cross_layer_1/StatefulPartitionedCallвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/StatefulPartitionedCallв%dense_layer_1/StatefulPartitionedCallв$embedding_26/StatefulPartitionedCallв$embedding_27/StatefulPartitionedCallв$embedding_28/StatefulPartitionedCallв$embedding_29/StatefulPartitionedCallв$embedding_30/StatefulPartitionedCallв$embedding_31/StatefulPartitionedCallв$embedding_32/StatefulPartitionedCallв$embedding_33/StatefulPartitionedCallв$embedding_34/StatefulPartitionedCallв$embedding_35/StatefulPartitionedCallв$embedding_36/StatefulPartitionedCallв$embedding_37/StatefulPartitionedCallв$embedding_38/StatefulPartitionedCallв$embedding_39/StatefulPartitionedCallв$embedding_40/StatefulPartitionedCallв$embedding_41/StatefulPartitionedCallв$embedding_42/StatefulPartitionedCallв$embedding_43/StatefulPartitionedCallв$embedding_44/StatefulPartitionedCallв$embedding_45/StatefulPartitionedCallв$embedding_46/StatefulPartitionedCallв$embedding_47/StatefulPartitionedCallв$embedding_48/StatefulPartitionedCallв$embedding_49/StatefulPartitionedCallв$embedding_50/StatefulPartitionedCallв$embedding_51/StatefulPartitionedCall{
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
strided_slice/stack_2Ў
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2А
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2д
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_64037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_621812&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3д
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_64044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_621992&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4д
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_64051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_622172&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5д
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_64058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_622352&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6д
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_64065*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_622532&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7д
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_64072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_622712&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8д
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_64079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_622892&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9д
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_64086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_623072&
$embedding_33/StatefulPartitionedCallБ
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10е
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_64093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_623252&
$embedding_34/StatefulPartitionedCallБ
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11е
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_64100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_623432&
$embedding_35/StatefulPartitionedCallБ
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12е
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_64107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_623612&
$embedding_36/StatefulPartitionedCallБ
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13е
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_64114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_623792&
$embedding_37/StatefulPartitionedCallБ
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14е
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_64121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_623972&
$embedding_38/StatefulPartitionedCallБ
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15е
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_64128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_624152&
$embedding_39/StatefulPartitionedCallБ
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16е
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_64135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_624332&
$embedding_40/StatefulPartitionedCallБ
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17е
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_64142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_624512&
$embedding_41/StatefulPartitionedCallБ
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18е
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_64149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_624692&
$embedding_42/StatefulPartitionedCallБ
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19е
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_64156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_624872&
$embedding_43/StatefulPartitionedCallБ
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20е
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_64163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_625052&
$embedding_44/StatefulPartitionedCallБ
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21е
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_64170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_625232&
$embedding_45/StatefulPartitionedCallБ
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22е
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_64177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_625412&
$embedding_46/StatefulPartitionedCallБ
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23е
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_64184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_625592&
$embedding_47/StatefulPartitionedCallБ
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24е
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_64191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_625772&
$embedding_48/StatefulPartitionedCallБ
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25е
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_64198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_625952&
$embedding_49/StatefulPartitionedCallБ
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26е
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_64205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_626132&
$embedding_50/StatefulPartitionedCallБ
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27е
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_64212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_626312&
$embedding_51/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┤

concatConcatV2-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1Э
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0batch_normalization_1_64219batch_normalization_1_64221batch_normalization_1_64223batch_normalization_1_64225*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620742/
-batch_normalization_1/StatefulPartitionedCallМ
%cross_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0cross_layer_1_64228cross_layer_1_64230cross_layer_1_64232cross_layer_1_64234*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_630142'
%cross_layer_1/StatefulPartitionedCall╣
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_layer_1_64237dense_layer_1_64239dense_layer_1_64241dense_layer_1_64243dense_layer_1_64245dense_layer_1_64247*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_629322'
%dense_layer_1/StatefulPartitionedCall`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╘
concat_2ConcatV2.cross_layer_1/StatefulPartitionedCall:output:0.dense_layer_1/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2Ъ
dense_7/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_7_64252dense_7_64254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_627592!
dense_7/StatefulPartitionedCally
SigmoidSigmoid(dense_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2	
Sigmoid¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_64230*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_64228*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_64232*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul√
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_64234*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity╣
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall&^cross_layer_1/StatefulPartitionedCallR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2N
%cross_layer_1/StatefulPartitionedCall%cross_layer_1/StatefulPartitionedCall2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
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
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╩
А
,__inference_embedding_30_layer_call_fn_65794

inputs
unknown:*
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_622532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╒
є
__inference_loss_fn_1_65687n
[dcn__attention__parallel__v1_1_cross_layer_1_wk0_regularizer_square_readvariableop_resource:	▌
identityИвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp┼
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOp[dcn__attention__parallel__v1_1_cross_layer_1_wk0_regularizer_square_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulО
IdentityIdentityDdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityг
NoOpNoOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp
Ё
■
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_62932

inputs9
&dense_4_matmul_readvariableop_resource:	▌ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02
dense_4/MatMul/ReadVariableOpЛ
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/BiasAdds
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityС
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╩
А
,__inference_embedding_47_layer_call_fn_66083

inputs
unknown:

identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_625592
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_50_layer_call_and_return_conditional_losses_66127

inputs(
embedding_lookup_66121:'
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66121Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66121*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66121*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_31_layer_call_fn_65811

inputs
unknown:	
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_622712
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
∙)
╒
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65386

inputs6
'assignmovingavg_readvariableop_resource:	▌8
)assignmovingavg_1_readvariableop_resource:	▌+
cast_readvariableop_resource:	▌-
cast_1_readvariableop_resource:	▌
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	▌2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ▌2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:▌*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Д
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:▌2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_30_layer_call_and_return_conditional_losses_65787

inputs(
embedding_lookup_65781:*
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65781Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65781*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65781*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_44_layer_call_fn_66032

inputs
unknown:	└
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_625052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_37_layer_call_fn_65913

inputs
unknown:	х
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_623792
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╖
╧

>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65332

inputs
unknown:h
	unknown_0:	 
	unknown_1:	╬
	unknown_2:	┼
	unknown_3:*
	unknown_4:	
	unknown_5:	╨
	unknown_6:C
	unknown_7:
	unknown_8:	╕
	unknown_9:	Т

unknown_10:	х

unknown_11:	┌

unknown_12:

unknown_13:	╪


unknown_14:	╔

unknown_15:


unknown_16:	Г

unknown_17:	└

unknown_18:

unknown_19:	З

unknown_20:


unknown_21:

unknown_22:	╕

unknown_23:'

unknown_24:	╩

unknown_25:	▌

unknown_26:	▌

unknown_27:	▌

unknown_28:	▌

unknown_29:	▌

unknown_30:	▌

unknown_31:	▌

unknown_32:	▌

unknown_33:	▌ 

unknown_34: 

unknown_35:  

unknown_36: 

unknown_37: 

unknown_38:

unknown_39:	▐

unknown_40:
identityИвStatefulPartitionedCall▒
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
:         *J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_635852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
Ф╡
Ў"
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64772

inputs5
#embedding_26_embedding_lookup_64420:h6
#embedding_27_embedding_lookup_64430:	 6
#embedding_28_embedding_lookup_64440:	╬6
#embedding_29_embedding_lookup_64450:	┼5
#embedding_30_embedding_lookup_64460:*5
#embedding_31_embedding_lookup_64470:	6
#embedding_32_embedding_lookup_64480:	╨5
#embedding_33_embedding_lookup_64490:C5
#embedding_34_embedding_lookup_64500:6
#embedding_35_embedding_lookup_64510:	╕6
#embedding_36_embedding_lookup_64520:	Т6
#embedding_37_embedding_lookup_64530:	х6
#embedding_38_embedding_lookup_64540:	┌5
#embedding_39_embedding_lookup_64550:6
#embedding_40_embedding_lookup_64560:	╪
6
#embedding_41_embedding_lookup_64570:	╔5
#embedding_42_embedding_lookup_64580:
6
#embedding_43_embedding_lookup_64590:	Г6
#embedding_44_embedding_lookup_64600:	└5
#embedding_45_embedding_lookup_64610:6
#embedding_46_embedding_lookup_64620:	З5
#embedding_47_embedding_lookup_64630:
5
#embedding_48_embedding_lookup_64640:6
#embedding_49_embedding_lookup_64650:	╕5
#embedding_50_embedding_lookup_64660:'6
#embedding_51_embedding_lookup_64670:	╩A
2batch_normalization_1_cast_readvariableop_resource:	▌C
4batch_normalization_1_cast_1_readvariableop_resource:	▌C
4batch_normalization_1_cast_2_readvariableop_resource:	▌C
4batch_normalization_1_cast_3_readvariableop_resource:	▌<
)cross_layer_1_mul_readvariableop_resource:	▌>
+cross_layer_1_mul_1_readvariableop_resource:	▌>
+cross_layer_1_mul_2_readvariableop_resource:	▌<
)cross_layer_1_add_readvariableop_resource:	▌G
4dense_layer_1_dense_4_matmul_readvariableop_resource:	▌ C
5dense_layer_1_dense_4_biasadd_readvariableop_resource: F
4dense_layer_1_dense_5_matmul_readvariableop_resource:  C
5dense_layer_1_dense_5_biasadd_readvariableop_resource: F
4dense_layer_1_dense_6_matmul_readvariableop_resource: C
5dense_layer_1_dense_6_biasadd_readvariableop_resource:9
&dense_7_matmul_readvariableop_resource:	▐5
'dense_7_biasadd_readvariableop_resource:
identityИв)batch_normalization_1/Cast/ReadVariableOpв+batch_normalization_1/Cast_1/ReadVariableOpв+batch_normalization_1/Cast_2/ReadVariableOpв+batch_normalization_1/Cast_3/ReadVariableOpв cross_layer_1/Mul/ReadVariableOpв"cross_layer_1/Mul_1/ReadVariableOpв"cross_layer_1/Mul_2/ReadVariableOpв cross_layer_1/add/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOpв,dense_layer_1/dense_4/BiasAdd/ReadVariableOpв+dense_layer_1/dense_4/MatMul/ReadVariableOpв,dense_layer_1/dense_5/BiasAdd/ReadVariableOpв+dense_layer_1/dense_5/MatMul/ReadVariableOpв,dense_layer_1/dense_6/BiasAdd/ReadVariableOpв+dense_layer_1/dense_6/MatMul/ReadVariableOpвembedding_26/embedding_lookupвembedding_27/embedding_lookupвembedding_28/embedding_lookupвembedding_29/embedding_lookupвembedding_30/embedding_lookupвembedding_31/embedding_lookupвembedding_32/embedding_lookupвembedding_33/embedding_lookupвembedding_34/embedding_lookupвembedding_35/embedding_lookupвembedding_36/embedding_lookupвembedding_37/embedding_lookupвembedding_38/embedding_lookupвembedding_39/embedding_lookupвembedding_40/embedding_lookupвembedding_41/embedding_lookupвembedding_42/embedding_lookupвembedding_43/embedding_lookupвembedding_44/embedding_lookupвembedding_45/embedding_lookupвembedding_46/embedding_lookupвembedding_47/embedding_lookupвembedding_48/embedding_lookupвembedding_49/embedding_lookupвembedding_50/embedding_lookupвembedding_51/embedding_lookup{
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
strided_slice/stack_2ї
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2 
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2Е
embedding_26/CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_26/Cast╕
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_64420embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/64420*'
_output_shapes
:         *
dtype02
embedding_26/embedding_lookupЬ
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/64420*'
_output_shapes
:         2(
&embedding_26/embedding_lookup/Identity├
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_26/embedding_lookup/Identity_1
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3Е
embedding_27/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_27/Cast╕
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_64430embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/64430*'
_output_shapes
:         *
dtype02
embedding_27/embedding_lookupЬ
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/64430*'
_output_shapes
:         2(
&embedding_27/embedding_lookup/Identity├
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_27/embedding_lookup/Identity_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4Е
embedding_28/CastCaststrided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_28/Cast╕
embedding_28/embedding_lookupResourceGather#embedding_28_embedding_lookup_64440embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_28/embedding_lookup/64440*'
_output_shapes
:         *
dtype02
embedding_28/embedding_lookupЬ
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_28/embedding_lookup/64440*'
_output_shapes
:         2(
&embedding_28/embedding_lookup/Identity├
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_28/embedding_lookup/Identity_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5Е
embedding_29/CastCaststrided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_29/Cast╕
embedding_29/embedding_lookupResourceGather#embedding_29_embedding_lookup_64450embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_29/embedding_lookup/64450*'
_output_shapes
:         *
dtype02
embedding_29/embedding_lookupЬ
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_29/embedding_lookup/64450*'
_output_shapes
:         2(
&embedding_29/embedding_lookup/Identity├
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_29/embedding_lookup/Identity_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6Е
embedding_30/CastCaststrided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_30/Cast╕
embedding_30/embedding_lookupResourceGather#embedding_30_embedding_lookup_64460embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_30/embedding_lookup/64460*'
_output_shapes
:         *
dtype02
embedding_30/embedding_lookupЬ
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_30/embedding_lookup/64460*'
_output_shapes
:         2(
&embedding_30/embedding_lookup/Identity├
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_30/embedding_lookup/Identity_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7Е
embedding_31/CastCaststrided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_31/Cast╕
embedding_31/embedding_lookupResourceGather#embedding_31_embedding_lookup_64470embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_31/embedding_lookup/64470*'
_output_shapes
:         *
dtype02
embedding_31/embedding_lookupЬ
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_31/embedding_lookup/64470*'
_output_shapes
:         2(
&embedding_31/embedding_lookup/Identity├
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_31/embedding_lookup/Identity_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8Е
embedding_32/CastCaststrided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_32/Cast╕
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_64480embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/64480*'
_output_shapes
:         *
dtype02
embedding_32/embedding_lookupЬ
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/64480*'
_output_shapes
:         2(
&embedding_32/embedding_lookup/Identity├
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_32/embedding_lookup/Identity_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9Е
embedding_33/CastCaststrided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_33/Cast╕
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_64490embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/64490*'
_output_shapes
:         *
dtype02
embedding_33/embedding_lookupЬ
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/64490*'
_output_shapes
:         2(
&embedding_33/embedding_lookup/Identity├
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_33/embedding_lookup/Identity_1Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10Ж
embedding_34/CastCaststrided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_34/Cast╕
embedding_34/embedding_lookupResourceGather#embedding_34_embedding_lookup_64500embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_34/embedding_lookup/64500*'
_output_shapes
:         *
dtype02
embedding_34/embedding_lookupЬ
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_34/embedding_lookup/64500*'
_output_shapes
:         2(
&embedding_34/embedding_lookup/Identity├
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_34/embedding_lookup/Identity_1Б
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11Ж
embedding_35/CastCaststrided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_35/Cast╕
embedding_35/embedding_lookupResourceGather#embedding_35_embedding_lookup_64510embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_35/embedding_lookup/64510*'
_output_shapes
:         *
dtype02
embedding_35/embedding_lookupЬ
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_35/embedding_lookup/64510*'
_output_shapes
:         2(
&embedding_35/embedding_lookup/Identity├
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_35/embedding_lookup/Identity_1Б
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12Ж
embedding_36/CastCaststrided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_36/Cast╕
embedding_36/embedding_lookupResourceGather#embedding_36_embedding_lookup_64520embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_36/embedding_lookup/64520*'
_output_shapes
:         *
dtype02
embedding_36/embedding_lookupЬ
&embedding_36/embedding_lookup/IdentityIdentity&embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_36/embedding_lookup/64520*'
_output_shapes
:         2(
&embedding_36/embedding_lookup/Identity├
(embedding_36/embedding_lookup/Identity_1Identity/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_36/embedding_lookup/Identity_1Б
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13Ж
embedding_37/CastCaststrided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_37/Cast╕
embedding_37/embedding_lookupResourceGather#embedding_37_embedding_lookup_64530embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_37/embedding_lookup/64530*'
_output_shapes
:         *
dtype02
embedding_37/embedding_lookupЬ
&embedding_37/embedding_lookup/IdentityIdentity&embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_37/embedding_lookup/64530*'
_output_shapes
:         2(
&embedding_37/embedding_lookup/Identity├
(embedding_37/embedding_lookup/Identity_1Identity/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_37/embedding_lookup/Identity_1Б
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14Ж
embedding_38/CastCaststrided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_38/Cast╕
embedding_38/embedding_lookupResourceGather#embedding_38_embedding_lookup_64540embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_38/embedding_lookup/64540*'
_output_shapes
:         *
dtype02
embedding_38/embedding_lookupЬ
&embedding_38/embedding_lookup/IdentityIdentity&embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_38/embedding_lookup/64540*'
_output_shapes
:         2(
&embedding_38/embedding_lookup/Identity├
(embedding_38/embedding_lookup/Identity_1Identity/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_38/embedding_lookup/Identity_1Б
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15Ж
embedding_39/CastCaststrided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_39/Cast╕
embedding_39/embedding_lookupResourceGather#embedding_39_embedding_lookup_64550embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_39/embedding_lookup/64550*'
_output_shapes
:         *
dtype02
embedding_39/embedding_lookupЬ
&embedding_39/embedding_lookup/IdentityIdentity&embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_39/embedding_lookup/64550*'
_output_shapes
:         2(
&embedding_39/embedding_lookup/Identity├
(embedding_39/embedding_lookup/Identity_1Identity/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_39/embedding_lookup/Identity_1Б
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16Ж
embedding_40/CastCaststrided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_40/Cast╕
embedding_40/embedding_lookupResourceGather#embedding_40_embedding_lookup_64560embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_40/embedding_lookup/64560*'
_output_shapes
:         *
dtype02
embedding_40/embedding_lookupЬ
&embedding_40/embedding_lookup/IdentityIdentity&embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_40/embedding_lookup/64560*'
_output_shapes
:         2(
&embedding_40/embedding_lookup/Identity├
(embedding_40/embedding_lookup/Identity_1Identity/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_40/embedding_lookup/Identity_1Б
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17Ж
embedding_41/CastCaststrided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_41/Cast╕
embedding_41/embedding_lookupResourceGather#embedding_41_embedding_lookup_64570embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_41/embedding_lookup/64570*'
_output_shapes
:         *
dtype02
embedding_41/embedding_lookupЬ
&embedding_41/embedding_lookup/IdentityIdentity&embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_41/embedding_lookup/64570*'
_output_shapes
:         2(
&embedding_41/embedding_lookup/Identity├
(embedding_41/embedding_lookup/Identity_1Identity/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_41/embedding_lookup/Identity_1Б
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18Ж
embedding_42/CastCaststrided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_42/Cast╕
embedding_42/embedding_lookupResourceGather#embedding_42_embedding_lookup_64580embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_42/embedding_lookup/64580*'
_output_shapes
:         *
dtype02
embedding_42/embedding_lookupЬ
&embedding_42/embedding_lookup/IdentityIdentity&embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_42/embedding_lookup/64580*'
_output_shapes
:         2(
&embedding_42/embedding_lookup/Identity├
(embedding_42/embedding_lookup/Identity_1Identity/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_42/embedding_lookup/Identity_1Б
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19Ж
embedding_43/CastCaststrided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_43/Cast╕
embedding_43/embedding_lookupResourceGather#embedding_43_embedding_lookup_64590embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_43/embedding_lookup/64590*'
_output_shapes
:         *
dtype02
embedding_43/embedding_lookupЬ
&embedding_43/embedding_lookup/IdentityIdentity&embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_43/embedding_lookup/64590*'
_output_shapes
:         2(
&embedding_43/embedding_lookup/Identity├
(embedding_43/embedding_lookup/Identity_1Identity/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_43/embedding_lookup/Identity_1Б
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20Ж
embedding_44/CastCaststrided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_44/Cast╕
embedding_44/embedding_lookupResourceGather#embedding_44_embedding_lookup_64600embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_44/embedding_lookup/64600*'
_output_shapes
:         *
dtype02
embedding_44/embedding_lookupЬ
&embedding_44/embedding_lookup/IdentityIdentity&embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_44/embedding_lookup/64600*'
_output_shapes
:         2(
&embedding_44/embedding_lookup/Identity├
(embedding_44/embedding_lookup/Identity_1Identity/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_44/embedding_lookup/Identity_1Б
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21Ж
embedding_45/CastCaststrided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_45/Cast╕
embedding_45/embedding_lookupResourceGather#embedding_45_embedding_lookup_64610embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_45/embedding_lookup/64610*'
_output_shapes
:         *
dtype02
embedding_45/embedding_lookupЬ
&embedding_45/embedding_lookup/IdentityIdentity&embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_45/embedding_lookup/64610*'
_output_shapes
:         2(
&embedding_45/embedding_lookup/Identity├
(embedding_45/embedding_lookup/Identity_1Identity/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_45/embedding_lookup/Identity_1Б
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22Ж
embedding_46/CastCaststrided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_46/Cast╕
embedding_46/embedding_lookupResourceGather#embedding_46_embedding_lookup_64620embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_46/embedding_lookup/64620*'
_output_shapes
:         *
dtype02
embedding_46/embedding_lookupЬ
&embedding_46/embedding_lookup/IdentityIdentity&embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_46/embedding_lookup/64620*'
_output_shapes
:         2(
&embedding_46/embedding_lookup/Identity├
(embedding_46/embedding_lookup/Identity_1Identity/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_46/embedding_lookup/Identity_1Б
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23Ж
embedding_47/CastCaststrided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_47/Cast╕
embedding_47/embedding_lookupResourceGather#embedding_47_embedding_lookup_64630embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_47/embedding_lookup/64630*'
_output_shapes
:         *
dtype02
embedding_47/embedding_lookupЬ
&embedding_47/embedding_lookup/IdentityIdentity&embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_47/embedding_lookup/64630*'
_output_shapes
:         2(
&embedding_47/embedding_lookup/Identity├
(embedding_47/embedding_lookup/Identity_1Identity/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_47/embedding_lookup/Identity_1Б
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24Ж
embedding_48/CastCaststrided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_48/Cast╕
embedding_48/embedding_lookupResourceGather#embedding_48_embedding_lookup_64640embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_48/embedding_lookup/64640*'
_output_shapes
:         *
dtype02
embedding_48/embedding_lookupЬ
&embedding_48/embedding_lookup/IdentityIdentity&embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_48/embedding_lookup/64640*'
_output_shapes
:         2(
&embedding_48/embedding_lookup/Identity├
(embedding_48/embedding_lookup/Identity_1Identity/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_48/embedding_lookup/Identity_1Б
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25Ж
embedding_49/CastCaststrided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_49/Cast╕
embedding_49/embedding_lookupResourceGather#embedding_49_embedding_lookup_64650embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_49/embedding_lookup/64650*'
_output_shapes
:         *
dtype02
embedding_49/embedding_lookupЬ
&embedding_49/embedding_lookup/IdentityIdentity&embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_49/embedding_lookup/64650*'
_output_shapes
:         2(
&embedding_49/embedding_lookup/Identity├
(embedding_49/embedding_lookup/Identity_1Identity/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_49/embedding_lookup/Identity_1Б
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26Ж
embedding_50/CastCaststrided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_50/Cast╕
embedding_50/embedding_lookupResourceGather#embedding_50_embedding_lookup_64660embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_50/embedding_lookup/64660*'
_output_shapes
:         *
dtype02
embedding_50/embedding_lookupЬ
&embedding_50/embedding_lookup/IdentityIdentity&embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_50/embedding_lookup/64660*'
_output_shapes
:         2(
&embedding_50/embedding_lookup/Identity├
(embedding_50/embedding_lookup/Identity_1Identity/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_50/embedding_lookup/Identity_1Б
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27Ж
embedding_51/CastCaststrided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_51/Cast╕
embedding_51/embedding_lookupResourceGather#embedding_51_embedding_lookup_64670embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_51/embedding_lookup/64670*'
_output_shapes
:         *
dtype02
embedding_51/embedding_lookupЬ
&embedding_51/embedding_lookup/IdentityIdentity&embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_51/embedding_lookup/64670*'
_output_shapes
:         2(
&embedding_51/embedding_lookup/Identity├
(embedding_51/embedding_lookup/Identity_1Identity/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_51/embedding_lookup/Identity_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЬ
concatConcatV21embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:01embedding_28/embedding_lookup/Identity_1:output:01embedding_29/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:01embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:01embedding_34/embedding_lookup/Identity_1:output:01embedding_35/embedding_lookup/Identity_1:output:01embedding_36/embedding_lookup/Identity_1:output:01embedding_37/embedding_lookup/Identity_1:output:01embedding_38/embedding_lookup/Identity_1:output:01embedding_39/embedding_lookup/Identity_1:output:01embedding_40/embedding_lookup/Identity_1:output:01embedding_41/embedding_lookup/Identity_1:output:01embedding_42/embedding_lookup/Identity_1:output:01embedding_43/embedding_lookup/Identity_1:output:01embedding_44/embedding_lookup/Identity_1:output:01embedding_45/embedding_lookup/Identity_1:output:01embedding_46/embedding_lookup/Identity_1:output:01embedding_47/embedding_lookup/Identity_1:output:01embedding_48/embedding_lookup/Identity_1:output:01embedding_49/embedding_lookup/Identity_1:output:01embedding_50/embedding_lookup/Identity_1:output:01embedding_51/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1╞
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:▌*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp╠
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOp╠
+batch_normalization_1/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:▌*
dtype02-
+batch_normalization_1/Cast_2/ReadVariableOp╠
+batch_normalization_1/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:▌*
dtype02-
+batch_normalization_1/Cast_3/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y▐
#batch_normalization_1/batchnorm/addAddV23batch_normalization_1/Cast_1/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/addж
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:▌2'
%batch_normalization_1/batchnorm/Rsqrt╫
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/mul─
%batch_normalization_1/batchnorm/mul_1Mulconcat_1:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2'
%batch_normalization_1/batchnorm/mul_1╫
%batch_normalization_1/batchnorm/mul_2Mul1batch_normalization_1/Cast/ReadVariableOp:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2'
%batch_normalization_1/batchnorm/mul_2╫
#batch_normalization_1/batchnorm/subSub3batch_normalization_1/Cast_2/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/sub▐
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2'
%batch_normalization_1/batchnorm/add_1~
cross_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
cross_layer_1/ExpandDims/dim╦
cross_layer_1/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0%cross_layer_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/ExpandDimsп
 cross_layer_1/Mul/ReadVariableOpReadVariableOp)cross_layer_1_mul_readvariableop_resource*
_output_shapes
:	▌*
dtype02"
 cross_layer_1/Mul/ReadVariableOp▒
cross_layer_1/MulMul!cross_layer_1/ExpandDims:output:0(cross_layer_1/Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul╡
"cross_layer_1/Mul_1/ReadVariableOpReadVariableOp+cross_layer_1_mul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02$
"cross_layer_1/Mul_1/ReadVariableOp╖
cross_layer_1/Mul_1Mul!cross_layer_1/ExpandDims:output:0*cross_layer_1/Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_1╡
"cross_layer_1/Mul_2/ReadVariableOpReadVariableOp+cross_layer_1_mul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02$
"cross_layer_1/Mul_2/ReadVariableOp╖
cross_layer_1/Mul_2Mul!cross_layer_1/ExpandDims:output:0*cross_layer_1/Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_2▓
cross_layer_1/MatMulBatchMatMulV2cross_layer_1/Mul:z:0cross_layer_1/Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
cross_layer_1/MatMulД
cross_layer_1/ExpExpcross_layer_1/MatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
cross_layer_1/ExpМ
#cross_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#cross_layer_1/Sum/reduction_indices║
cross_layer_1/SumSumcross_layer_1/Exp:y:0,cross_layer_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
cross_layer_1/Sumд
cross_layer_1/truedivRealDivcross_layer_1/Exp:y:0cross_layer_1/Sum:output:0*
T0*-
_output_shapes
:         ▌▌2
cross_layer_1/truediv╢
cross_layer_1/MatMul_1BatchMatMulV2cross_layer_1/truediv:z:0!cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/MatMul_1в
cross_layer_1/Mul_3Mulcross_layer_1/Mul_2:z:0cross_layer_1/MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_3п
 cross_layer_1/add/ReadVariableOpReadVariableOp)cross_layer_1_add_readvariableop_resource*
_output_shapes
:	▌*
dtype02"
 cross_layer_1/add/ReadVariableOpй
cross_layer_1/addAddV2cross_layer_1/Mul_3:z:0(cross_layer_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/addд
cross_layer_1/add_1AddV2cross_layer_1/add:z:0!cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/add_1Ь
cross_layer_1/SqueezeSqueezecross_layer_1/add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2
cross_layer_1/Squeeze╨
+dense_layer_1/dense_4/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02-
+dense_layer_1/dense_4/MatMul/ReadVariableOp╪
dense_layer_1/dense_4/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:03dense_layer_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/MatMul╬
,dense_layer_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,dense_layer_1/dense_4/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_4/BiasAddBiasAdd&dense_layer_1/dense_4/MatMul:product:04dense_layer_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/BiasAddЪ
dense_layer_1/dense_4/ReluRelu&dense_layer_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/Relu╧
+dense_layer_1/dense_5/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+dense_layer_1/dense_5/MatMul/ReadVariableOp╫
dense_layer_1/dense_5/MatMulMatMul(dense_layer_1/dense_4/Relu:activations:03dense_layer_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/MatMul╬
,dense_layer_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,dense_layer_1/dense_5/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_5/BiasAddBiasAdd&dense_layer_1/dense_5/MatMul:product:04dense_layer_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/BiasAddЪ
dense_layer_1/dense_5/ReluRelu&dense_layer_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/Relu╧
+dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+dense_layer_1/dense_6/MatMul/ReadVariableOp╫
dense_layer_1/dense_6/MatMulMatMul(dense_layer_1/dense_5/Relu:activations:03dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_layer_1/dense_6/MatMul╬
,dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_6/BiasAddBiasAdd&dense_layer_1/dense_6/MatMul:product:04dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_layer_1/dense_6/BiasAdd`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╝
concat_2ConcatV2cross_layer_1/Squeeze:output:0&dense_layer_1/dense_6/BiasAdd:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2ж
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	▐*
dtype02
dense_7/MatMul/ReadVariableOpЦ
dense_7/MatMulMatMulconcat_2:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAddi
SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidХ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOp+cross_layer_1_mul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulУ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOp)cross_layer_1_mul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulХ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOp+cross_layer_1_mul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulС
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOp)cross_layer_1_add_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity 
NoOpNoOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp,^batch_normalization_1/Cast_2/ReadVariableOp,^batch_normalization_1/Cast_3/ReadVariableOp!^cross_layer_1/Mul/ReadVariableOp#^cross_layer_1/Mul_1/ReadVariableOp#^cross_layer_1/Mul_2/ReadVariableOp!^cross_layer_1/add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp-^dense_layer_1/dense_4/BiasAdd/ReadVariableOp,^dense_layer_1/dense_4/MatMul/ReadVariableOp-^dense_layer_1/dense_5/BiasAdd/ReadVariableOp,^dense_layer_1/dense_5/MatMul/ReadVariableOp-^dense_layer_1/dense_6/BiasAdd/ReadVariableOp,^dense_layer_1/dense_6/MatMul/ReadVariableOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup^embedding_36/embedding_lookup^embedding_37/embedding_lookup^embedding_38/embedding_lookup^embedding_39/embedding_lookup^embedding_40/embedding_lookup^embedding_41/embedding_lookup^embedding_42/embedding_lookup^embedding_43/embedding_lookup^embedding_44/embedding_lookup^embedding_45/embedding_lookup^embedding_46/embedding_lookup^embedding_47/embedding_lookup^embedding_48/embedding_lookup^embedding_49/embedding_lookup^embedding_50/embedding_lookup^embedding_51/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2Z
+batch_normalization_1/Cast_2/ReadVariableOp+batch_normalization_1/Cast_2/ReadVariableOp2Z
+batch_normalization_1/Cast_3/ReadVariableOp+batch_normalization_1/Cast_3/ReadVariableOp2D
 cross_layer_1/Mul/ReadVariableOp cross_layer_1/Mul/ReadVariableOp2H
"cross_layer_1/Mul_1/ReadVariableOp"cross_layer_1/Mul_1/ReadVariableOp2H
"cross_layer_1/Mul_2/ReadVariableOp"cross_layer_1/Mul_2/ReadVariableOp2D
 cross_layer_1/add/ReadVariableOp cross_layer_1/add/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2\
,dense_layer_1/dense_4/BiasAdd/ReadVariableOp,dense_layer_1/dense_4/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_4/MatMul/ReadVariableOp+dense_layer_1/dense_4/MatMul/ReadVariableOp2\
,dense_layer_1/dense_5/BiasAdd/ReadVariableOp,dense_layer_1/dense_5/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_5/MatMul/ReadVariableOp+dense_layer_1/dense_5/MatMul/ReadVariableOp2\
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp,dense_layer_1/dense_6/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_6/MatMul/ReadVariableOp+dense_layer_1/dense_6/MatMul/ReadVariableOp2>
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
embedding_51/embedding_lookupembedding_51/embedding_lookup:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_46_layer_call_fn_66066

inputs
unknown:	З
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_625412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╖В
╜
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64022
input_1$
embedding_26_63776:h%
embedding_27_63783:	 %
embedding_28_63790:	╬%
embedding_29_63797:	┼$
embedding_30_63804:*$
embedding_31_63811:	%
embedding_32_63818:	╨$
embedding_33_63825:C$
embedding_34_63832:%
embedding_35_63839:	╕%
embedding_36_63846:	Т%
embedding_37_63853:	х%
embedding_38_63860:	┌$
embedding_39_63867:%
embedding_40_63874:	╪
%
embedding_41_63881:	╔$
embedding_42_63888:
%
embedding_43_63895:	Г%
embedding_44_63902:	└$
embedding_45_63909:%
embedding_46_63916:	З$
embedding_47_63923:
$
embedding_48_63930:%
embedding_49_63937:	╕$
embedding_50_63944:'%
embedding_51_63951:	╩*
batch_normalization_1_63958:	▌*
batch_normalization_1_63960:	▌*
batch_normalization_1_63962:	▌*
batch_normalization_1_63964:	▌&
cross_layer_1_63967:	▌&
cross_layer_1_63969:	▌&
cross_layer_1_63971:	▌&
cross_layer_1_63973:	▌&
dense_layer_1_63976:	▌ !
dense_layer_1_63978: %
dense_layer_1_63980:  !
dense_layer_1_63982: %
dense_layer_1_63984: !
dense_layer_1_63986: 
dense_7_63991:	▐
dense_7_63993:
identityИв-batch_normalization_1/StatefulPartitionedCallв%cross_layer_1/StatefulPartitionedCallвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/StatefulPartitionedCallв%dense_layer_1/StatefulPartitionedCallв$embedding_26/StatefulPartitionedCallв$embedding_27/StatefulPartitionedCallв$embedding_28/StatefulPartitionedCallв$embedding_29/StatefulPartitionedCallв$embedding_30/StatefulPartitionedCallв$embedding_31/StatefulPartitionedCallв$embedding_32/StatefulPartitionedCallв$embedding_33/StatefulPartitionedCallв$embedding_34/StatefulPartitionedCallв$embedding_35/StatefulPartitionedCallв$embedding_36/StatefulPartitionedCallв$embedding_37/StatefulPartitionedCallв$embedding_38/StatefulPartitionedCallв$embedding_39/StatefulPartitionedCallв$embedding_40/StatefulPartitionedCallв$embedding_41/StatefulPartitionedCallв$embedding_42/StatefulPartitionedCallв$embedding_43/StatefulPartitionedCallв$embedding_44/StatefulPartitionedCallв$embedding_45/StatefulPartitionedCallв$embedding_46/StatefulPartitionedCallв$embedding_47/StatefulPartitionedCallв$embedding_48/StatefulPartitionedCallв$embedding_49/StatefulPartitionedCallв$embedding_50/StatefulPartitionedCallв$embedding_51/StatefulPartitionedCall{
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
strided_slice/stack_2Ў
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2А
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2д
$embedding_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_26_63776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_26_layer_call_and_return_conditional_losses_621812&
$embedding_26/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3д
$embedding_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_27_63783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_621992&
$embedding_27/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4д
$embedding_28/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_28_63790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_622172&
$embedding_28/StatefulPartitionedCall
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5д
$embedding_29/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_5:output:0embedding_29_63797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_29_layer_call_and_return_conditional_losses_622352&
$embedding_29/StatefulPartitionedCall
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6д
$embedding_30/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_6:output:0embedding_30_63804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_30_layer_call_and_return_conditional_losses_622532&
$embedding_30/StatefulPartitionedCall
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7д
$embedding_31/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_7:output:0embedding_31_63811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_31_layer_call_and_return_conditional_losses_622712&
$embedding_31/StatefulPartitionedCall
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8д
$embedding_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_8:output:0embedding_32_63818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_622892&
$embedding_32/StatefulPartitionedCall
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9д
$embedding_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_9:output:0embedding_33_63825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_623072&
$embedding_33/StatefulPartitionedCallБ
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10е
$embedding_34/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_10:output:0embedding_34_63832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_34_layer_call_and_return_conditional_losses_623252&
$embedding_34/StatefulPartitionedCallБ
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11е
$embedding_35/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_11:output:0embedding_35_63839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_623432&
$embedding_35/StatefulPartitionedCallБ
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12е
$embedding_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_12:output:0embedding_36_63846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_36_layer_call_and_return_conditional_losses_623612&
$embedding_36/StatefulPartitionedCallБ
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13е
$embedding_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_13:output:0embedding_37_63853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_37_layer_call_and_return_conditional_losses_623792&
$embedding_37/StatefulPartitionedCallБ
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14е
$embedding_38/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_14:output:0embedding_38_63860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_623972&
$embedding_38/StatefulPartitionedCallБ
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15е
$embedding_39/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_15:output:0embedding_39_63867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_39_layer_call_and_return_conditional_losses_624152&
$embedding_39/StatefulPartitionedCallБ
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16е
$embedding_40/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_16:output:0embedding_40_63874*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_624332&
$embedding_40/StatefulPartitionedCallБ
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17е
$embedding_41/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_17:output:0embedding_41_63881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_41_layer_call_and_return_conditional_losses_624512&
$embedding_41/StatefulPartitionedCallБ
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18е
$embedding_42/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_18:output:0embedding_42_63888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_42_layer_call_and_return_conditional_losses_624692&
$embedding_42/StatefulPartitionedCallБ
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19е
$embedding_43/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_19:output:0embedding_43_63895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_624872&
$embedding_43/StatefulPartitionedCallБ
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20е
$embedding_44/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_20:output:0embedding_44_63902*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_44_layer_call_and_return_conditional_losses_625052&
$embedding_44/StatefulPartitionedCallБ
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21е
$embedding_45/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_21:output:0embedding_45_63909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_625232&
$embedding_45/StatefulPartitionedCallБ
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22е
$embedding_46/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_22:output:0embedding_46_63916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_46_layer_call_and_return_conditional_losses_625412&
$embedding_46/StatefulPartitionedCallБ
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23е
$embedding_47/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_23:output:0embedding_47_63923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_47_layer_call_and_return_conditional_losses_625592&
$embedding_47/StatefulPartitionedCallБ
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24е
$embedding_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_24:output:0embedding_48_63930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_625772&
$embedding_48/StatefulPartitionedCallБ
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25е
$embedding_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_25:output:0embedding_49_63937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_625952&
$embedding_49/StatefulPartitionedCallБ
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26е
$embedding_50/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_26:output:0embedding_50_63944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_626132&
$embedding_50/StatefulPartitionedCallБ
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27е
$embedding_51/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_27:output:0embedding_51_63951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_51_layer_call_and_return_conditional_losses_626312&
$embedding_51/StatefulPartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis┤

concatConcatV2-embedding_26/StatefulPartitionedCall:output:0-embedding_27/StatefulPartitionedCall:output:0-embedding_28/StatefulPartitionedCall:output:0-embedding_29/StatefulPartitionedCall:output:0-embedding_30/StatefulPartitionedCall:output:0-embedding_31/StatefulPartitionedCall:output:0-embedding_32/StatefulPartitionedCall:output:0-embedding_33/StatefulPartitionedCall:output:0-embedding_34/StatefulPartitionedCall:output:0-embedding_35/StatefulPartitionedCall:output:0-embedding_36/StatefulPartitionedCall:output:0-embedding_37/StatefulPartitionedCall:output:0-embedding_38/StatefulPartitionedCall:output:0-embedding_39/StatefulPartitionedCall:output:0-embedding_40/StatefulPartitionedCall:output:0-embedding_41/StatefulPartitionedCall:output:0-embedding_42/StatefulPartitionedCall:output:0-embedding_43/StatefulPartitionedCall:output:0-embedding_44/StatefulPartitionedCall:output:0-embedding_45/StatefulPartitionedCall:output:0-embedding_46/StatefulPartitionedCall:output:0-embedding_47/StatefulPartitionedCall:output:0-embedding_48/StatefulPartitionedCall:output:0-embedding_49/StatefulPartitionedCall:output:0-embedding_50/StatefulPartitionedCall:output:0-embedding_51/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1Я
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0batch_normalization_1_63958batch_normalization_1_63960batch_normalization_1_63962batch_normalization_1_63964*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620142/
-batch_normalization_1/StatefulPartitionedCallМ
%cross_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0cross_layer_1_63967cross_layer_1_63969cross_layer_1_63971cross_layer_1_63973*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_626992'
%cross_layer_1/StatefulPartitionedCall╣
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0dense_layer_1_63976dense_layer_1_63978dense_layer_1_63980dense_layer_1_63982dense_layer_1_63984dense_layer_1_63986*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_627332'
%dense_layer_1/StatefulPartitionedCall`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╘
concat_2ConcatV2.cross_layer_1/StatefulPartitionedCall:output:0.dense_layer_1/StatefulPartitionedCall:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2Ъ
dense_7/StatefulPartitionedCallStatefulPartitionedCallconcat_2:output:0dense_7_63991dense_7_63993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_627592!
dense_7/StatefulPartitionedCally
SigmoidSigmoid(dense_7/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2	
Sigmoid¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63969*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63967*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul¤
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63971*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul√
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpcross_layer_1_63973*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity╣
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall&^cross_layer_1/StatefulPartitionedCallR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp ^dense_7/StatefulPartitionedCall&^dense_layer_1/StatefulPartitionedCall%^embedding_26/StatefulPartitionedCall%^embedding_27/StatefulPartitionedCall%^embedding_28/StatefulPartitionedCall%^embedding_29/StatefulPartitionedCall%^embedding_30/StatefulPartitionedCall%^embedding_31/StatefulPartitionedCall%^embedding_32/StatefulPartitionedCall%^embedding_33/StatefulPartitionedCall%^embedding_34/StatefulPartitionedCall%^embedding_35/StatefulPartitionedCall%^embedding_36/StatefulPartitionedCall%^embedding_37/StatefulPartitionedCall%^embedding_38/StatefulPartitionedCall%^embedding_39/StatefulPartitionedCall%^embedding_40/StatefulPartitionedCall%^embedding_41/StatefulPartitionedCall%^embedding_42/StatefulPartitionedCall%^embedding_43/StatefulPartitionedCall%^embedding_44/StatefulPartitionedCall%^embedding_45/StatefulPartitionedCall%^embedding_46/StatefulPartitionedCall%^embedding_47/StatefulPartitionedCall%^embedding_48/StatefulPartitionedCall%^embedding_49/StatefulPartitionedCall%^embedding_50/StatefulPartitionedCall%^embedding_51/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2N
%cross_layer_1/StatefulPartitionedCall%cross_layer_1/StatefulPartitionedCall2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
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
$embedding_51/StatefulPartitionedCall$embedding_51/StatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╦
Б
,__inference_embedding_28_layer_call_fn_65760

inputs
unknown:	╬
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_28_layer_call_and_return_conditional_losses_622172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_49_layer_call_fn_66117

inputs
unknown:	╕
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_49_layer_call_and_return_conditional_losses_625952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
█W
╓
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65620

inputs.
mul_readvariableop_resource:	▌0
mul_1_readvariableop_resource:	▌0
mul_2_readvariableop_resource:	▌.
add_readvariableop_resource:	▌
identityИвMul/ReadVariableOpвMul_1/ReadVariableOpвMul_2/ReadVariableOpвadd/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2

ExpandDimsЕ
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul/ReadVariableOpy
MulMulExpandDims:output:0Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
MulЛ
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_1/ReadVariableOp
Mul_1MulExpandDims:output:0Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_1Л
Mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_2/ReadVariableOp
Mul_2MulExpandDims:output:0Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_2z
MatMulBatchMatMulV2Mul:z:0	Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
MatMulZ
ExpExpMatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
Expp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesВ
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
Suml
truedivRealDivExp:y:0Sum:output:0*
T0*-
_output_shapes
:         ▌▌2	
truediv~
MatMul_1BatchMatMulV2truediv:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2

MatMul_1j
Mul_3Mul	Mul_2:z:0MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
Mul_3Е
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02
add/ReadVariableOpq
addAddV2	Mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
addl
add_1AddV2add:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
add_1r
SqueezeSqueeze	add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2	
SqueezeЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulЕ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulГ
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mull
IdentityIdentitySqueeze:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identity∙
NoOpNoOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp^Mul_2/ReadVariableOp^add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2,
Mul_2/ReadVariableOpMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_48_layer_call_and_return_conditional_losses_62577

inputs(
embedding_lookup_62571:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62571Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62571*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62571*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_43_layer_call_fn_66015

inputs
unknown:	Г
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_43_layer_call_and_return_conditional_losses_624872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_48_layer_call_fn_66100

inputs
unknown:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_48_layer_call_and_return_conditional_losses_625772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╝
╨

>__inference_dcn__attention__parallel__v1_1_layer_call_fn_62878
input_1
unknown:h
	unknown_0:	 
	unknown_1:	╬
	unknown_2:	┼
	unknown_3:*
	unknown_4:	
	unknown_5:	╨
	unknown_6:C
	unknown_7:
	unknown_8:	╕
	unknown_9:	Т

unknown_10:	х

unknown_11:	┌

unknown_12:

unknown_13:	╪


unknown_14:	╔

unknown_15:


unknown_16:	Г

unknown_17:	└

unknown_18:

unknown_19:	З

unknown_20:


unknown_21:

unknown_22:	╕

unknown_23:'

unknown_24:	╩

unknown_25:	▌

unknown_26:	▌

unknown_27:	▌

unknown_28:	▌

unknown_29:	▌

unknown_30:	▌

unknown_31:	▌

unknown_32:	▌

unknown_33:	▌ 

unknown_34: 

unknown_35:  

unknown_36: 

unknown_37: 

unknown_38:

unknown_39:	▐

unknown_40:
identityИвStatefulPartitionedCall┤
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
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_627912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
╠

д
G__inference_embedding_32_layer_call_and_return_conditional_losses_65821

inputs)
embedding_lookup_65815:	╨
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65815Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65815*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65815*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_38_layer_call_and_return_conditional_losses_62397

inputs)
embedding_lookup_62391:	┌
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62391Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62391*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62391*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_50_layer_call_and_return_conditional_losses_62613

inputs(
embedding_lookup_62607:'
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62607Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62607*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62607*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
и

Ї
B__inference_dense_7_layer_call_and_return_conditional_losses_62759

inputs1
matmul_readvariableop_resource:	▐-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	▐*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▐: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ▐
 
_user_specified_nameinputs
╠

д
G__inference_embedding_27_layer_call_and_return_conditional_losses_65736

inputs)
embedding_lookup_65730:	 
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65730Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65730*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65730*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_31_layer_call_and_return_conditional_losses_62271

inputs(
embedding_lookup_62265:	
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62265Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62265*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62265*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_38_layer_call_fn_65930

inputs
unknown:	┌
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_38_layer_call_and_return_conditional_losses_623972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_32_layer_call_fn_65828

inputs
unknown:	╨
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_32_layer_call_and_return_conditional_losses_622892
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╣
╧

>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65243

inputs
unknown:h
	unknown_0:	 
	unknown_1:	╬
	unknown_2:	┼
	unknown_3:*
	unknown_4:	
	unknown_5:	╨
	unknown_6:C
	unknown_7:
	unknown_8:	╕
	unknown_9:	Т

unknown_10:	х

unknown_11:	┌

unknown_12:

unknown_13:	╪


unknown_14:	╔

unknown_15:


unknown_16:	Г

unknown_17:	└

unknown_18:

unknown_19:	З

unknown_20:


unknown_21:

unknown_22:	╕

unknown_23:'

unknown_24:	╩

unknown_25:	▌

unknown_26:	▌

unknown_27:	▌

unknown_28:	▌

unknown_29:	▌

unknown_30:	▌

unknown_31:	▌

unknown_32:	▌

unknown_33:	▌ 

unknown_34: 

unknown_35:  

unknown_36: 

unknown_37: 

unknown_38:

unknown_39:	▐

unknown_40:
identityИвStatefulPartitionedCall│
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
:         *L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8В *b
f]R[
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_627912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╠

д
G__inference_embedding_29_layer_call_and_return_conditional_losses_62235

inputs)
embedding_lookup_62229:	┼
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62229Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62229*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62229*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
и

Ї
B__inference_dense_7_layer_call_and_return_conditional_losses_65656

inputs1
matmul_readvariableop_resource:	▐-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	▐*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ▐: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ▐
 
_user_specified_nameinputs
╦

г
G__inference_embedding_34_layer_call_and_return_conditional_losses_62325

inputs(
embedding_lookup_62319:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62319Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62319*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62319*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_28_layer_call_and_return_conditional_losses_65753

inputs)
embedding_lookup_65747:	╬
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65747Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65747*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65747*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╠

д
G__inference_embedding_37_layer_call_and_return_conditional_losses_62379

inputs)
embedding_lookup_62373:	х
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62373Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62373*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62373*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_33_layer_call_fn_65845

inputs
unknown:C
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_33_layer_call_and_return_conditional_losses_623072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
Ё
■
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_62733

inputs9
&dense_4_matmul_readvariableop_resource:	▌ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02
dense_4/MatMul/ReadVariableOpЛ
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/BiasAdds
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityС
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╠

д
G__inference_embedding_36_layer_call_and_return_conditional_losses_65889

inputs)
embedding_lookup_65883:	Т
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65883Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65883*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65883*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╒
є
__inference_loss_fn_2_65698n
[dcn__attention__parallel__v1_1_cross_layer_1_wv0_regularizer_square_readvariableop_resource:	▌
identityИвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp┼
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOp[dcn__attention__parallel__v1_1_cross_layer_1_wv0_regularizer_square_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulО
IdentityIdentityDdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

Identityг
NoOpNoOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp
╦
Б
,__inference_embedding_40_layer_call_fn_65964

inputs
unknown:	╪

identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_40_layer_call_and_return_conditional_losses_624332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
∙)
╒
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_62074

inputs6
'assignmovingavg_readvariableop_resource:	▌8
)assignmovingavg_1_readvariableop_resource:	▌+
cast_readvariableop_resource:	▌-
cast_1_readvariableop_resource:	▌
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвCast/ReadVariableOpвCast_1/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	▌2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         ▌2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayе
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:▌*
dtype02 
AssignMovingAvg/ReadVariableOpЩ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg/subР
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg/mul┐
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
╫#<2
AssignMovingAvg_1/decayл
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02"
 AssignMovingAvg_1/ReadVariableOpб
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg_1/subШ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:▌2
AssignMovingAvg_1/mul╔
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1Д
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast/ReadVariableOpК
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02
Cast_1/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:▌2
batchnorm/Rsqrt
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2
batchnorm/mul_2}
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2
batchnorm/add_1o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityц
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╠

д
G__inference_embedding_32_layer_call_and_return_conditional_losses_62289

inputs)
embedding_lookup_62283:	╨
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62283Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62283*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62283*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
█W
╓
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_62699

inputs.
mul_readvariableop_resource:	▌0
mul_1_readvariableop_resource:	▌0
mul_2_readvariableop_resource:	▌.
add_readvariableop_resource:	▌
identityИвMul/ReadVariableOpвMul_1/ReadVariableOpвMul_2/ReadVariableOpвadd/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2

ExpandDimsЕ
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul/ReadVariableOpy
MulMulExpandDims:output:0Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
MulЛ
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_1/ReadVariableOp
Mul_1MulExpandDims:output:0Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_1Л
Mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_2/ReadVariableOp
Mul_2MulExpandDims:output:0Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_2z
MatMulBatchMatMulV2Mul:z:0	Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
MatMulZ
ExpExpMatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
Expp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesВ
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
Suml
truedivRealDivExp:y:0Sum:output:0*
T0*-
_output_shapes
:         ▌▌2	
truediv~
MatMul_1BatchMatMulV2truediv:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2

MatMul_1j
Mul_3Mul	Mul_2:z:0MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
Mul_3Е
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02
add/ReadVariableOpq
addAddV2	Mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
addl
add_1AddV2add:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
add_1r
SqueezeSqueeze	add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2	
SqueezeЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulЕ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulГ
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mull
IdentityIdentitySqueeze:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identity∙
NoOpNoOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp^Mul_2/ReadVariableOp^add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2,
Mul_2/ReadVariableOpMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
р
╘
5__inference_batch_normalization_1_layer_call_fn_65399

inputs
unknown:	▌
	unknown_0:	▌
	unknown_1:	▌
	unknown_2:	▌
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_620142
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_48_layer_call_and_return_conditional_losses_66093

inputs(
embedding_lookup_66087:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66087Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66087*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66087*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_50_layer_call_fn_66134

inputs
unknown:'
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_50_layer_call_and_return_conditional_losses_626132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
р
▄
-__inference_cross_layer_1_layer_call_fn_65633

inputs
unknown:	▌
	unknown_0:	▌
	unknown_1:	▌
	unknown_2:	▌
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ▌*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_626992
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╠

д
G__inference_embedding_37_layer_call_and_return_conditional_losses_65906

inputs)
embedding_lookup_65900:	х
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65900Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65900*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65900*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╙╣
╠\
__inference__traced_save_66579
file_prefixY
Usavev2_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_read_readvariableopX
Tsavev2_dcn__attention__parallel__v1_1_batch_normalization_1_beta_read_readvariableop_
[savev2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_mean_read_readvariableopc
_savev2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_variance_read_readvariableopO
Ksavev2_dcn__attention__parallel__v1_1_cross_layer_1_wq0_read_readvariableopO
Ksavev2_dcn__attention__parallel__v1_1_cross_layer_1_wk0_read_readvariableopO
Ksavev2_dcn__attention__parallel__v1_1_cross_layer_1_wv0_read_readvariableopN
Jsavev2_dcn__attention__parallel__v1_1_cross_layer_1_b0_read_readvariableopL
Hsavev2_dcn__attention__parallel__v1_1_dense_7_kernel_read_readvariableopJ
Fsavev2_dcn__attention__parallel__v1_1_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_26_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_27_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_36_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_37_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_38_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_39_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_40_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_41_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_42_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_43_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_44_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_45_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_28_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_46_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_47_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_48_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_49_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_50_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_51_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_29_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_30_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_31_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_32_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_33_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_34_embeddings_read_readvariableopU
Qsavev2_dcn__attention__parallel__v1_1_embedding_35_embeddings_read_readvariableopZ
Vsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_read_readvariableopX
Tsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_read_readvariableopZ
Vsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_read_readvariableopX
Tsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_read_readvariableopZ
Vsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_read_readvariableopX
Tsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop`
\savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_m_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_m_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_m_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_m_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_m_read_readvariableopU
Qsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_m_read_readvariableopS
Osavev2_adam_dcn__attention__parallel__v1_1_dense_7_kernel_m_read_readvariableopQ
Msavev2_adam_dcn__attention__parallel__v1_1_dense_7_bias_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_m_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_m_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_m_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_m_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_m_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_m_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_m_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_m_read_readvariableop`
\savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_v_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_v_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_v_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_v_read_readvariableopV
Rsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_v_read_readvariableopU
Qsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_v_read_readvariableopS
Osavev2_adam_dcn__attention__parallel__v1_1_dense_7_kernel_v_read_readvariableopQ
Msavev2_adam_dcn__attention__parallel__v1_1_dense_7_bias_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_v_read_readvariableop\
Xsavev2_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_v_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_v_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_v_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_v_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_v_read_readvariableopa
]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_v_read_readvariableop_
[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╧?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:И*
dtype0*р>
value╓>B╙>ИB#bn/gamma/.ATTRIBUTES/VARIABLE_VALUEB"bn/beta/.ATTRIBUTES/VARIABLE_VALUEB)bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB-bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wq0/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wk0/.ATTRIBUTES/VARIABLE_VALUEB*cross_layer/wv0/.ATTRIBUTES/VARIABLE_VALUEB)cross_layer/b0/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?bn/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB>bn/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wq0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wk0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFcross_layer/wv0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEcross_layer/b0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJoutput_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHoutput_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЭ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:И*
dtype0*ж
valueЬBЩИB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesКZ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Usavev2_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_read_readvariableopTsavev2_dcn__attention__parallel__v1_1_batch_normalization_1_beta_read_readvariableop[savev2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_mean_read_readvariableop_savev2_dcn__attention__parallel__v1_1_batch_normalization_1_moving_variance_read_readvariableopKsavev2_dcn__attention__parallel__v1_1_cross_layer_1_wq0_read_readvariableopKsavev2_dcn__attention__parallel__v1_1_cross_layer_1_wk0_read_readvariableopKsavev2_dcn__attention__parallel__v1_1_cross_layer_1_wv0_read_readvariableopJsavev2_dcn__attention__parallel__v1_1_cross_layer_1_b0_read_readvariableopHsavev2_dcn__attention__parallel__v1_1_dense_7_kernel_read_readvariableopFsavev2_dcn__attention__parallel__v1_1_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_26_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_27_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_36_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_37_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_38_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_39_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_40_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_41_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_42_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_43_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_44_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_45_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_28_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_46_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_47_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_48_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_49_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_50_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_51_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_29_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_30_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_31_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_32_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_33_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_34_embeddings_read_readvariableopQsavev2_dcn__attention__parallel__v1_1_embedding_35_embeddings_read_readvariableopVsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_read_readvariableopTsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_read_readvariableopVsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_read_readvariableopTsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_read_readvariableopVsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_read_readvariableopTsavev2_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop\savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_m_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_m_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_m_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_m_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_m_read_readvariableopQsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_m_read_readvariableopOsavev2_adam_dcn__attention__parallel__v1_1_dense_7_kernel_m_read_readvariableopMsavev2_adam_dcn__attention__parallel__v1_1_dense_7_bias_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_m_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_m_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_m_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_m_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_m_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_m_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_m_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_m_read_readvariableop\savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_gamma_v_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_batch_normalization_1_beta_v_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wq0_v_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wk0_v_read_readvariableopRsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_wv0_v_read_readvariableopQsavev2_adam_dcn__attention__parallel__v1_1_cross_layer_1_b0_v_read_readvariableopOsavev2_adam_dcn__attention__parallel__v1_1_dense_7_kernel_v_read_readvariableopMsavev2_adam_dcn__attention__parallel__v1_1_dense_7_bias_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_26_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_27_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_36_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_37_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_38_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_39_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_40_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_41_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_42_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_43_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_44_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_45_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_28_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_46_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_47_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_48_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_49_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_50_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_51_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_29_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_30_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_31_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_32_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_33_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_34_embeddings_v_read_readvariableopXsavev2_adam_dcn__attention__parallel__v1_1_embedding_35_embeddings_v_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_kernel_v_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_4_bias_v_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_kernel_v_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_5_bias_v_read_readvariableop]savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_kernel_v_read_readvariableop[savev2_adam_dcn__attention__parallel__v1_1_dense_layer_1_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Щ
dtypesО
Л2И	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*К

_input_shapes°	
ї	: :▌:▌:▌:▌:	▌:	▌:	▌:	▌:	▐:: : : : : :h:	 :	Т:	х:	┌::	╪
:	╔:
:	Г:	└::	╬:	З:
::	╕:':	╩:	┼:*:	:	╨:C::	╕:	▌ : :  : : :: : :╚:╚:╚:╚:::▌:▌:	▌:	▌:	▌:	▌:	▐::h:	 :	Т:	х:	┌::	╪
:	╔:
:	Г:	└::	╬:	З:
::	╕:':	╩:	┼:*:	:	╨:C::	╕:	▌ : :  : : ::▌:▌:	▌:	▌:	▌:	▌:	▐::h:	 :	Т:	х:	┌::	╪
:	╔:
:	Г:	└::	╬:	З:
::	╕:':	╩:	┼:*:	:	╨:C::	╕:	▌ : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:▌:!

_output_shapes	
:▌:!

_output_shapes	
:▌:!

_output_shapes	
:▌:%!

_output_shapes
:	▌:%!

_output_shapes
:	▌:%!

_output_shapes
:	▌:%!

_output_shapes
:	▌:%	!

_output_shapes
:	▐: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:h:%!

_output_shapes
:	 :%!

_output_shapes
:	Т:%!

_output_shapes
:	х:%!

_output_shapes
:	┌:$ 

_output_shapes

::%!

_output_shapes
:	╪
:%!

_output_shapes
:	╔:$ 

_output_shapes

:
:%!

_output_shapes
:	Г:%!

_output_shapes
:	└:$ 

_output_shapes

::%!

_output_shapes
:	╬:%!

_output_shapes
:	З:$ 

_output_shapes

:
:$ 

_output_shapes

::% !

_output_shapes
:	╕:$! 

_output_shapes

:':%"!

_output_shapes
:	╩:%#!

_output_shapes
:	┼:$$ 

_output_shapes

:*:$% 

_output_shapes

:	:%&!

_output_shapes
:	╨:$' 

_output_shapes

:C:$( 

_output_shapes

::%)!

_output_shapes
:	╕:%*!

_output_shapes
:	▌ : +

_output_shapes
: :$, 

_output_shapes

:  : -

_output_shapes
: :$. 

_output_shapes

: : /
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
:╚:!3

_output_shapes	
:╚:!4

_output_shapes	
:╚:!5

_output_shapes	
:╚: 6

_output_shapes
:: 7

_output_shapes
::!8

_output_shapes	
:▌:!9

_output_shapes	
:▌:%:!

_output_shapes
:	▌:%;!

_output_shapes
:	▌:%<!

_output_shapes
:	▌:%=!

_output_shapes
:	▌:%>!

_output_shapes
:	▐: ?

_output_shapes
::$@ 

_output_shapes

:h:%A!

_output_shapes
:	 :%B!

_output_shapes
:	Т:%C!

_output_shapes
:	х:%D!

_output_shapes
:	┌:$E 

_output_shapes

::%F!

_output_shapes
:	╪
:%G!

_output_shapes
:	╔:$H 

_output_shapes

:
:%I!

_output_shapes
:	Г:%J!

_output_shapes
:	└:$K 

_output_shapes

::%L!

_output_shapes
:	╬:%M!

_output_shapes
:	З:$N 

_output_shapes

:
:$O 

_output_shapes

::%P!

_output_shapes
:	╕:$Q 

_output_shapes

:':%R!

_output_shapes
:	╩:%S!

_output_shapes
:	┼:$T 

_output_shapes

:*:$U 

_output_shapes

:	:%V!

_output_shapes
:	╨:$W 

_output_shapes

:C:$X 

_output_shapes

::%Y!

_output_shapes
:	╕:%Z!

_output_shapes
:	▌ : [

_output_shapes
: :$\ 

_output_shapes

:  : ]

_output_shapes
: :$^ 

_output_shapes

: : _

_output_shapes
::!`

_output_shapes	
:▌:!a

_output_shapes	
:▌:%b!

_output_shapes
:	▌:%c!

_output_shapes
:	▌:%d!

_output_shapes
:	▌:%e!

_output_shapes
:	▌:%f!

_output_shapes
:	▐: g

_output_shapes
::$h 

_output_shapes

:h:%i!

_output_shapes
:	 :%j!

_output_shapes
:	Т:%k!

_output_shapes
:	х:%l!

_output_shapes
:	┌:$m 

_output_shapes

::%n!

_output_shapes
:	╪
:%o!

_output_shapes
:	╔:$p 

_output_shapes

:
:%q!

_output_shapes
:	Г:%r!

_output_shapes
:	└:$s 

_output_shapes

::%t!

_output_shapes
:	╬:%u!

_output_shapes
:	З:$v 

_output_shapes

:
:$w 

_output_shapes

::%x!

_output_shapes
:	╕:$y 

_output_shapes

:':%z!

_output_shapes
:	╩:%{!

_output_shapes
:	┼:$| 

_output_shapes

:*:$} 

_output_shapes

:	:%~!

_output_shapes
:	╨:$ 

_output_shapes

:C:%А 

_output_shapes

::&Б!

_output_shapes
:	╕:&В!

_output_shapes
:	▌ :!Г

_output_shapes
: :%Д 

_output_shapes

:  :!Е

_output_shapes
: :%Ж 

_output_shapes

: :!З

_output_shapes
::И

_output_shapes
: 
╠

д
G__inference_embedding_29_layer_call_and_return_conditional_losses_65770

inputs)
embedding_lookup_65764:	┼
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65764Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65764*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65764*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
РО
Ч4
 __inference__wrapped_model_61990
input_1T
Bdcn__attention__parallel__v1_1_embedding_26_embedding_lookup_61662:hU
Bdcn__attention__parallel__v1_1_embedding_27_embedding_lookup_61672:	 U
Bdcn__attention__parallel__v1_1_embedding_28_embedding_lookup_61682:	╬U
Bdcn__attention__parallel__v1_1_embedding_29_embedding_lookup_61692:	┼T
Bdcn__attention__parallel__v1_1_embedding_30_embedding_lookup_61702:*T
Bdcn__attention__parallel__v1_1_embedding_31_embedding_lookup_61712:	U
Bdcn__attention__parallel__v1_1_embedding_32_embedding_lookup_61722:	╨T
Bdcn__attention__parallel__v1_1_embedding_33_embedding_lookup_61732:CT
Bdcn__attention__parallel__v1_1_embedding_34_embedding_lookup_61742:U
Bdcn__attention__parallel__v1_1_embedding_35_embedding_lookup_61752:	╕U
Bdcn__attention__parallel__v1_1_embedding_36_embedding_lookup_61762:	ТU
Bdcn__attention__parallel__v1_1_embedding_37_embedding_lookup_61772:	хU
Bdcn__attention__parallel__v1_1_embedding_38_embedding_lookup_61782:	┌T
Bdcn__attention__parallel__v1_1_embedding_39_embedding_lookup_61792:U
Bdcn__attention__parallel__v1_1_embedding_40_embedding_lookup_61802:	╪
U
Bdcn__attention__parallel__v1_1_embedding_41_embedding_lookup_61812:	╔T
Bdcn__attention__parallel__v1_1_embedding_42_embedding_lookup_61822:
U
Bdcn__attention__parallel__v1_1_embedding_43_embedding_lookup_61832:	ГU
Bdcn__attention__parallel__v1_1_embedding_44_embedding_lookup_61842:	└T
Bdcn__attention__parallel__v1_1_embedding_45_embedding_lookup_61852:U
Bdcn__attention__parallel__v1_1_embedding_46_embedding_lookup_61862:	ЗT
Bdcn__attention__parallel__v1_1_embedding_47_embedding_lookup_61872:
T
Bdcn__attention__parallel__v1_1_embedding_48_embedding_lookup_61882:U
Bdcn__attention__parallel__v1_1_embedding_49_embedding_lookup_61892:	╕T
Bdcn__attention__parallel__v1_1_embedding_50_embedding_lookup_61902:'U
Bdcn__attention__parallel__v1_1_embedding_51_embedding_lookup_61912:	╩`
Qdcn__attention__parallel__v1_1_batch_normalization_1_cast_readvariableop_resource:	▌b
Sdcn__attention__parallel__v1_1_batch_normalization_1_cast_1_readvariableop_resource:	▌b
Sdcn__attention__parallel__v1_1_batch_normalization_1_cast_2_readvariableop_resource:	▌b
Sdcn__attention__parallel__v1_1_batch_normalization_1_cast_3_readvariableop_resource:	▌[
Hdcn__attention__parallel__v1_1_cross_layer_1_mul_readvariableop_resource:	▌]
Jdcn__attention__parallel__v1_1_cross_layer_1_mul_1_readvariableop_resource:	▌]
Jdcn__attention__parallel__v1_1_cross_layer_1_mul_2_readvariableop_resource:	▌[
Hdcn__attention__parallel__v1_1_cross_layer_1_add_readvariableop_resource:	▌f
Sdcn__attention__parallel__v1_1_dense_layer_1_dense_4_matmul_readvariableop_resource:	▌ b
Tdcn__attention__parallel__v1_1_dense_layer_1_dense_4_biasadd_readvariableop_resource: e
Sdcn__attention__parallel__v1_1_dense_layer_1_dense_5_matmul_readvariableop_resource:  b
Tdcn__attention__parallel__v1_1_dense_layer_1_dense_5_biasadd_readvariableop_resource: e
Sdcn__attention__parallel__v1_1_dense_layer_1_dense_6_matmul_readvariableop_resource: b
Tdcn__attention__parallel__v1_1_dense_layer_1_dense_6_biasadd_readvariableop_resource:X
Edcn__attention__parallel__v1_1_dense_7_matmul_readvariableop_resource:	▐T
Fdcn__attention__parallel__v1_1_dense_7_biasadd_readvariableop_resource:
identityИвHdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOpвJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOpвJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOpвJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOpв?dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOpвAdcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOpвAdcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOpв?dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOpв=dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOpв<dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOpвKdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOpвJdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOpвKdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOpвJdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOpвKdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOpвJdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOpв<dcn__attention__parallel__v1_1/embedding_26/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_27/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_28/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_29/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_30/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_31/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_32/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_33/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_34/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_35/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_36/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_37/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_38/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_39/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_40/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_41/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_42/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_43/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_44/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_45/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_46/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_47/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_48/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_49/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_50/embedding_lookupв<dcn__attention__parallel__v1_1/embedding_51/embedding_lookup╣
2dcn__attention__parallel__v1_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        24
2dcn__attention__parallel__v1_1/strided_slice/stack╜
4dcn__attention__parallel__v1_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice/stack_1╜
4dcn__attention__parallel__v1_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      26
4dcn__attention__parallel__v1_1/strided_slice/stack_2С
,dcn__attention__parallel__v1_1/strided_sliceStridedSliceinput_1;dcn__attention__parallel__v1_1/strided_slice/stack:output:0=dcn__attention__parallel__v1_1/strided_slice/stack_1:output:0=dcn__attention__parallel__v1_1/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2.
,dcn__attention__parallel__v1_1/strided_slice╜
4dcn__attention__parallel__v1_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_1/stack┴
6dcn__attention__parallel__v1_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6dcn__attention__parallel__v1_1/strided_slice_1/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_1/stack_2Ы
.dcn__attention__parallel__v1_1/strided_slice_1StridedSliceinput_1=dcn__attention__parallel__v1_1/strided_slice_1/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_1/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask20
.dcn__attention__parallel__v1_1/strided_slice_1╜
4dcn__attention__parallel__v1_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        26
4dcn__attention__parallel__v1_1/strided_slice_2/stack┴
6dcn__attention__parallel__v1_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_2/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_2/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_2StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_2/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_2/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_2т
0dcn__attention__parallel__v1_1/embedding_26/CastCast7dcn__attention__parallel__v1_1/strided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_26/Cast╙
<dcn__attention__parallel__v1_1/embedding_26/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_26_embedding_lookup_616624dcn__attention__parallel__v1_1/embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_26/embedding_lookup/61662*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_26/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_26/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_26/embedding_lookup/61662*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_26/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_26/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_26/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_3/stack┴
6dcn__attention__parallel__v1_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_3/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_3/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_3StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_3/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_3/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_3т
0dcn__attention__parallel__v1_1/embedding_27/CastCast7dcn__attention__parallel__v1_1/strided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_27/Cast╙
<dcn__attention__parallel__v1_1/embedding_27/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_27_embedding_lookup_616724dcn__attention__parallel__v1_1/embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_27/embedding_lookup/61672*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_27/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_27/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_27/embedding_lookup/61672*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_27/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_27/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_27/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_4/stack┴
6dcn__attention__parallel__v1_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_4/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_4/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_4StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_4/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_4/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_4т
0dcn__attention__parallel__v1_1/embedding_28/CastCast7dcn__attention__parallel__v1_1/strided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_28/Cast╙
<dcn__attention__parallel__v1_1/embedding_28/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_28_embedding_lookup_616824dcn__attention__parallel__v1_1/embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_28/embedding_lookup/61682*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_28/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_28/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_28/embedding_lookup/61682*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_28/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_28/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_28/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_5/stack┴
6dcn__attention__parallel__v1_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_5/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_5/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_5StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_5/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_5/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_5т
0dcn__attention__parallel__v1_1/embedding_29/CastCast7dcn__attention__parallel__v1_1/strided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_29/Cast╙
<dcn__attention__parallel__v1_1/embedding_29/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_29_embedding_lookup_616924dcn__attention__parallel__v1_1/embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_29/embedding_lookup/61692*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_29/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_29/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_29/embedding_lookup/61692*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_29/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_29/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_29/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_6/stack┴
6dcn__attention__parallel__v1_1/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_6/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_6/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_6StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_6/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_6/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_6т
0dcn__attention__parallel__v1_1/embedding_30/CastCast7dcn__attention__parallel__v1_1/strided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_30/Cast╙
<dcn__attention__parallel__v1_1/embedding_30/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_30_embedding_lookup_617024dcn__attention__parallel__v1_1/embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_30/embedding_lookup/61702*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_30/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_30/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_30/embedding_lookup/61702*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_30/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_30/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_30/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_7/stack┴
6dcn__attention__parallel__v1_1/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_7/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_7/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_7StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_7/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_7/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_7т
0dcn__attention__parallel__v1_1/embedding_31/CastCast7dcn__attention__parallel__v1_1/strided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_31/Cast╙
<dcn__attention__parallel__v1_1/embedding_31/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_31_embedding_lookup_617124dcn__attention__parallel__v1_1/embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_31/embedding_lookup/61712*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_31/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_31/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_31/embedding_lookup/61712*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_31/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_31/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_31/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_8/stack┴
6dcn__attention__parallel__v1_1/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_8/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_8/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_8StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_8/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_8/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_8т
0dcn__attention__parallel__v1_1/embedding_32/CastCast7dcn__attention__parallel__v1_1/strided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_32/Cast╙
<dcn__attention__parallel__v1_1/embedding_32/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_32_embedding_lookup_617224dcn__attention__parallel__v1_1/embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_32/embedding_lookup/61722*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_32/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_32/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_32/embedding_lookup/61722*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_32/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_32/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_32/embedding_lookup/Identity_1╜
4dcn__attention__parallel__v1_1/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       26
4dcn__attention__parallel__v1_1/strided_slice_9/stack┴
6dcn__attention__parallel__v1_1/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       28
6dcn__attention__parallel__v1_1/strided_slice_9/stack_1┴
6dcn__attention__parallel__v1_1/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6dcn__attention__parallel__v1_1/strided_slice_9/stack_2▀
.dcn__attention__parallel__v1_1/strided_slice_9StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0=dcn__attention__parallel__v1_1/strided_slice_9/stack:output:0?dcn__attention__parallel__v1_1/strided_slice_9/stack_1:output:0?dcn__attention__parallel__v1_1/strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask20
.dcn__attention__parallel__v1_1/strided_slice_9т
0dcn__attention__parallel__v1_1/embedding_33/CastCast7dcn__attention__parallel__v1_1/strided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_33/Cast╙
<dcn__attention__parallel__v1_1/embedding_33/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_33_embedding_lookup_617324dcn__attention__parallel__v1_1/embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_33/embedding_lookup/61732*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_33/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_33/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_33/embedding_lookup/61732*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_33/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_33/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_33/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_10/stack├
7dcn__attention__parallel__v1_1/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   29
7dcn__attention__parallel__v1_1/strided_slice_10/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_10/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_10StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_10/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_10/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_10у
0dcn__attention__parallel__v1_1/embedding_34/CastCast8dcn__attention__parallel__v1_1/strided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_34/Cast╙
<dcn__attention__parallel__v1_1/embedding_34/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_34_embedding_lookup_617424dcn__attention__parallel__v1_1/embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_34/embedding_lookup/61742*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_34/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_34/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_34/embedding_lookup/61742*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_34/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_34/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_34/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   27
5dcn__attention__parallel__v1_1/strided_slice_11/stack├
7dcn__attention__parallel__v1_1/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   29
7dcn__attention__parallel__v1_1/strided_slice_11/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_11/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_11StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_11/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_11/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_11у
0dcn__attention__parallel__v1_1/embedding_35/CastCast8dcn__attention__parallel__v1_1/strided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_35/Cast╙
<dcn__attention__parallel__v1_1/embedding_35/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_35_embedding_lookup_617524dcn__attention__parallel__v1_1/embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_35/embedding_lookup/61752*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_35/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_35/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_35/embedding_lookup/61752*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_35/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_35/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_35/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   27
5dcn__attention__parallel__v1_1/strided_slice_12/stack├
7dcn__attention__parallel__v1_1/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_12/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_12/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_12StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_12/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_12/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_12у
0dcn__attention__parallel__v1_1/embedding_36/CastCast8dcn__attention__parallel__v1_1/strided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_36/Cast╙
<dcn__attention__parallel__v1_1/embedding_36/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_36_embedding_lookup_617624dcn__attention__parallel__v1_1/embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_36/embedding_lookup/61762*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_36/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_36/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_36/embedding_lookup/61762*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_36/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_36/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_36/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_13/stack├
7dcn__attention__parallel__v1_1/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_13/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_13/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_13StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_13/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_13/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_13у
0dcn__attention__parallel__v1_1/embedding_37/CastCast8dcn__attention__parallel__v1_1/strided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_37/Cast╙
<dcn__attention__parallel__v1_1/embedding_37/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_37_embedding_lookup_617724dcn__attention__parallel__v1_1/embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_37/embedding_lookup/61772*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_37/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_37/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_37/embedding_lookup/61772*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_37/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_37/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_37/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_14/stack├
7dcn__attention__parallel__v1_1/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_14/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_14/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_14StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_14/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_14/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_14у
0dcn__attention__parallel__v1_1/embedding_38/CastCast8dcn__attention__parallel__v1_1/strided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_38/Cast╙
<dcn__attention__parallel__v1_1/embedding_38/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_38_embedding_lookup_617824dcn__attention__parallel__v1_1/embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_38/embedding_lookup/61782*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_38/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_38/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_38/embedding_lookup/61782*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_38/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_38/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_38/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_15/stack├
7dcn__attention__parallel__v1_1/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_15/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_15/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_15StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_15/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_15/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_15у
0dcn__attention__parallel__v1_1/embedding_39/CastCast8dcn__attention__parallel__v1_1/strided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_39/Cast╙
<dcn__attention__parallel__v1_1/embedding_39/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_39_embedding_lookup_617924dcn__attention__parallel__v1_1/embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_39/embedding_lookup/61792*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_39/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_39/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_39/embedding_lookup/61792*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_39/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_39/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_39/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_16/stack├
7dcn__attention__parallel__v1_1/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_16/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_16/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_16StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_16/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_16/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_16у
0dcn__attention__parallel__v1_1/embedding_40/CastCast8dcn__attention__parallel__v1_1/strided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_40/Cast╙
<dcn__attention__parallel__v1_1/embedding_40/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_40_embedding_lookup_618024dcn__attention__parallel__v1_1/embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_40/embedding_lookup/61802*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_40/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_40/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_40/embedding_lookup/61802*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_40/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_40/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_40/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_17/stack├
7dcn__attention__parallel__v1_1/strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_17/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_17/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_17StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_17/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_17/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_17у
0dcn__attention__parallel__v1_1/embedding_41/CastCast8dcn__attention__parallel__v1_1/strided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_41/Cast╙
<dcn__attention__parallel__v1_1/embedding_41/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_41_embedding_lookup_618124dcn__attention__parallel__v1_1/embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_41/embedding_lookup/61812*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_41/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_41/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_41/embedding_lookup/61812*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_41/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_41/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_41/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_18/stack├
7dcn__attention__parallel__v1_1/strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_18/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_18/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_18StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_18/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_18/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_18у
0dcn__attention__parallel__v1_1/embedding_42/CastCast8dcn__attention__parallel__v1_1/strided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_42/Cast╙
<dcn__attention__parallel__v1_1/embedding_42/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_42_embedding_lookup_618224dcn__attention__parallel__v1_1/embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_42/embedding_lookup/61822*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_42/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_42/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_42/embedding_lookup/61822*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_42/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_42/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_42/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_19/stack├
7dcn__attention__parallel__v1_1/strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_19/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_19/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_19StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_19/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_19/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_19у
0dcn__attention__parallel__v1_1/embedding_43/CastCast8dcn__attention__parallel__v1_1/strided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_43/Cast╙
<dcn__attention__parallel__v1_1/embedding_43/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_43_embedding_lookup_618324dcn__attention__parallel__v1_1/embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_43/embedding_lookup/61832*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_43/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_43/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_43/embedding_lookup/61832*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_43/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_43/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_43/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_20/stack├
7dcn__attention__parallel__v1_1/strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_20/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_20/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_20StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_20/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_20/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_20у
0dcn__attention__parallel__v1_1/embedding_44/CastCast8dcn__attention__parallel__v1_1/strided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_44/Cast╙
<dcn__attention__parallel__v1_1/embedding_44/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_44_embedding_lookup_618424dcn__attention__parallel__v1_1/embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_44/embedding_lookup/61842*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_44/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_44/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_44/embedding_lookup/61842*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_44/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_44/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_44/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_21/stack├
7dcn__attention__parallel__v1_1/strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_21/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_21/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_21StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_21/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_21/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_21у
0dcn__attention__parallel__v1_1/embedding_45/CastCast8dcn__attention__parallel__v1_1/strided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_45/Cast╙
<dcn__attention__parallel__v1_1/embedding_45/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_45_embedding_lookup_618524dcn__attention__parallel__v1_1/embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_45/embedding_lookup/61852*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_45/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_45/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_45/embedding_lookup/61852*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_45/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_45/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_45/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_22/stack├
7dcn__attention__parallel__v1_1/strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_22/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_22/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_22StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_22/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_22/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_22у
0dcn__attention__parallel__v1_1/embedding_46/CastCast8dcn__attention__parallel__v1_1/strided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_46/Cast╙
<dcn__attention__parallel__v1_1/embedding_46/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_46_embedding_lookup_618624dcn__attention__parallel__v1_1/embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_46/embedding_lookup/61862*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_46/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_46/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_46/embedding_lookup/61862*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_46/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_46/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_46/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_23/stack├
7dcn__attention__parallel__v1_1/strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_23/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_23/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_23StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_23/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_23/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_23у
0dcn__attention__parallel__v1_1/embedding_47/CastCast8dcn__attention__parallel__v1_1/strided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_47/Cast╙
<dcn__attention__parallel__v1_1/embedding_47/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_47_embedding_lookup_618724dcn__attention__parallel__v1_1/embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_47/embedding_lookup/61872*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_47/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_47/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_47/embedding_lookup/61872*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_47/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_47/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_47/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_24/stack├
7dcn__attention__parallel__v1_1/strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_24/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_24/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_24StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_24/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_24/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_24у
0dcn__attention__parallel__v1_1/embedding_48/CastCast8dcn__attention__parallel__v1_1/strided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_48/Cast╙
<dcn__attention__parallel__v1_1/embedding_48/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_48_embedding_lookup_618824dcn__attention__parallel__v1_1/embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_48/embedding_lookup/61882*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_48/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_48/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_48/embedding_lookup/61882*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_48/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_48/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_48/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_25/stack├
7dcn__attention__parallel__v1_1/strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_25/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_25/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_25StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_25/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_25/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_25у
0dcn__attention__parallel__v1_1/embedding_49/CastCast8dcn__attention__parallel__v1_1/strided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_49/Cast╙
<dcn__attention__parallel__v1_1/embedding_49/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_49_embedding_lookup_618924dcn__attention__parallel__v1_1/embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_49/embedding_lookup/61892*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_49/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_49/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_49/embedding_lookup/61892*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_49/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_49/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_49/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_26/stack├
7dcn__attention__parallel__v1_1/strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_26/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_26/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_26StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_26/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_26/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_26у
0dcn__attention__parallel__v1_1/embedding_50/CastCast8dcn__attention__parallel__v1_1/strided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_50/Cast╙
<dcn__attention__parallel__v1_1/embedding_50/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_50_embedding_lookup_619024dcn__attention__parallel__v1_1/embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_50/embedding_lookup/61902*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_50/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_50/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_50/embedding_lookup/61902*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_50/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_50/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_50/embedding_lookup/Identity_1┐
5dcn__attention__parallel__v1_1/strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       27
5dcn__attention__parallel__v1_1/strided_slice_27/stack├
7dcn__attention__parallel__v1_1/strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       29
7dcn__attention__parallel__v1_1/strided_slice_27/stack_1├
7dcn__attention__parallel__v1_1/strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7dcn__attention__parallel__v1_1/strided_slice_27/stack_2ф
/dcn__attention__parallel__v1_1/strided_slice_27StridedSlice7dcn__attention__parallel__v1_1/strided_slice_1:output:0>dcn__attention__parallel__v1_1/strided_slice_27/stack:output:0@dcn__attention__parallel__v1_1/strided_slice_27/stack_1:output:0@dcn__attention__parallel__v1_1/strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask21
/dcn__attention__parallel__v1_1/strided_slice_27у
0dcn__attention__parallel__v1_1/embedding_51/CastCast8dcn__attention__parallel__v1_1/strided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:         22
0dcn__attention__parallel__v1_1/embedding_51/Cast╙
<dcn__attention__parallel__v1_1/embedding_51/embedding_lookupResourceGatherBdcn__attention__parallel__v1_1_embedding_51_embedding_lookup_619124dcn__attention__parallel__v1_1/embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_51/embedding_lookup/61912*'
_output_shapes
:         *
dtype02>
<dcn__attention__parallel__v1_1/embedding_51/embedding_lookupШ
Edcn__attention__parallel__v1_1/embedding_51/embedding_lookup/IdentityIdentityEdcn__attention__parallel__v1_1/embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*U
_classK
IGloc:@dcn__attention__parallel__v1_1/embedding_51/embedding_lookup/61912*'
_output_shapes
:         2G
Edcn__attention__parallel__v1_1/embedding_51/embedding_lookup/Identityа
Gdcn__attention__parallel__v1_1/embedding_51/embedding_lookup/Identity_1IdentityNdcn__attention__parallel__v1_1/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2I
Gdcn__attention__parallel__v1_1/embedding_51/embedding_lookup/Identity_1Ъ
*dcn__attention__parallel__v1_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*dcn__attention__parallel__v1_1/concat/axisЯ
%dcn__attention__parallel__v1_1/concatConcatV2Pdcn__attention__parallel__v1_1/embedding_26/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_27/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_28/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_29/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_30/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_31/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_32/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_33/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_34/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_35/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_36/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_37/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_38/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_39/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_40/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_41/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_42/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_43/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_44/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_45/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_46/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_47/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_48/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_49/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_50/embedding_lookup/Identity_1:output:0Pdcn__attention__parallel__v1_1/embedding_51/embedding_lookup/Identity_1:output:03dcn__attention__parallel__v1_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2'
%dcn__attention__parallel__v1_1/concatЮ
,dcn__attention__parallel__v1_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,dcn__attention__parallel__v1_1/concat_1/axis╕
'dcn__attention__parallel__v1_1/concat_1ConcatV25dcn__attention__parallel__v1_1/strided_slice:output:0.dcn__attention__parallel__v1_1/concat:output:05dcn__attention__parallel__v1_1/concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2)
'dcn__attention__parallel__v1_1/concat_1г
Hdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOpQdcn__attention__parallel__v1_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:▌*
dtype02J
Hdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOpй
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02L
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOpй
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:▌*
dtype02L
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOpй
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:▌*
dtype02L
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOp╤
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2F
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add/y┌
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/addAddV2Rdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0Mdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2D
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/addГ
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/RsqrtRsqrtFdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:▌2F
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/Rsqrt╙
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mulMulHdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/Rsqrt:y:0Rdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2D
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul└
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_1Mul0dcn__attention__parallel__v1_1/concat_1:output:0Fdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2F
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_1╙
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_2MulPdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOp:value:0Fdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2F
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_2╙
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/subSubRdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOp:value:0Hdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2D
Bdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/sub┌
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add_1AddV2Hdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/mul_1:z:0Fdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2F
Ddcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add_1╝
;dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2=
;dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims/dim╟
7dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims
ExpandDimsHdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add_1:z:0Ddcn__attention__parallel__v1_1/cross_layer_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌29
7dcn__attention__parallel__v1_1/cross_layer_1/ExpandDimsМ
?dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOpReadVariableOpHdcn__attention__parallel__v1_1_cross_layer_1_mul_readvariableop_resource*
_output_shapes
:	▌*
dtype02A
?dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOpн
0dcn__attention__parallel__v1_1/cross_layer_1/MulMul@dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims:output:0Gdcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌22
0dcn__attention__parallel__v1_1/cross_layer_1/MulТ
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOpReadVariableOpJdcn__attention__parallel__v1_1_cross_layer_1_mul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02C
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOp│
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_1Mul@dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims:output:0Idcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌24
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_1Т
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOpReadVariableOpJdcn__attention__parallel__v1_1_cross_layer_1_mul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02C
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOp│
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_2Mul@dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims:output:0Idcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌24
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_2о
3dcn__attention__parallel__v1_1/cross_layer_1/MatMulBatchMatMulV24dcn__attention__parallel__v1_1/cross_layer_1/Mul:z:06dcn__attention__parallel__v1_1/cross_layer_1/Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(25
3dcn__attention__parallel__v1_1/cross_layer_1/MatMulс
0dcn__attention__parallel__v1_1/cross_layer_1/ExpExp<dcn__attention__parallel__v1_1/cross_layer_1/MatMul:output:0*
T0*-
_output_shapes
:         ▌▌22
0dcn__attention__parallel__v1_1/cross_layer_1/Exp╩
Bdcn__attention__parallel__v1_1/cross_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2D
Bdcn__attention__parallel__v1_1/cross_layer_1/Sum/reduction_indices╢
0dcn__attention__parallel__v1_1/cross_layer_1/SumSum4dcn__attention__parallel__v1_1/cross_layer_1/Exp:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(22
0dcn__attention__parallel__v1_1/cross_layer_1/Sumа
4dcn__attention__parallel__v1_1/cross_layer_1/truedivRealDiv4dcn__attention__parallel__v1_1/cross_layer_1/Exp:y:09dcn__attention__parallel__v1_1/cross_layer_1/Sum:output:0*
T0*-
_output_shapes
:         ▌▌26
4dcn__attention__parallel__v1_1/cross_layer_1/truediv▓
5dcn__attention__parallel__v1_1/cross_layer_1/MatMul_1BatchMatMulV28dcn__attention__parallel__v1_1/cross_layer_1/truediv:z:0@dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌27
5dcn__attention__parallel__v1_1/cross_layer_1/MatMul_1Ю
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_3Mul6dcn__attention__parallel__v1_1/cross_layer_1/Mul_2:z:0>dcn__attention__parallel__v1_1/cross_layer_1/MatMul_1:output:0*
T0*,
_output_shapes
:         ▌24
2dcn__attention__parallel__v1_1/cross_layer_1/Mul_3М
?dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOpReadVariableOpHdcn__attention__parallel__v1_1_cross_layer_1_add_readvariableop_resource*
_output_shapes
:	▌*
dtype02A
?dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOpе
0dcn__attention__parallel__v1_1/cross_layer_1/addAddV26dcn__attention__parallel__v1_1/cross_layer_1/Mul_3:z:0Gdcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌22
0dcn__attention__parallel__v1_1/cross_layer_1/addа
2dcn__attention__parallel__v1_1/cross_layer_1/add_1AddV24dcn__attention__parallel__v1_1/cross_layer_1/add:z:0@dcn__attention__parallel__v1_1/cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌24
2dcn__attention__parallel__v1_1/cross_layer_1/add_1∙
4dcn__attention__parallel__v1_1/cross_layer_1/SqueezeSqueeze6dcn__attention__parallel__v1_1/cross_layer_1/add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
26
4dcn__attention__parallel__v1_1/cross_layer_1/Squeezeн
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_dense_layer_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02L
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOp╘
;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMulMatMulHdcn__attention__parallel__v1_1/batch_normalization_1/batchnorm/add_1:z:0Rdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2=
;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMulл
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOpReadVariableOpTdcn__attention__parallel__v1_1_dense_layer_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02M
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOp╒
<dcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAddBiasAddEdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul:product:0Sdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2>
<dcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAddў
9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/ReluReluEdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2;
9dcn__attention__parallel__v1_1/dense_layer_1/dense_4/Reluм
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_dense_layer_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02L
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOp╙
;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMulMatMulGdcn__attention__parallel__v1_1/dense_layer_1/dense_4/Relu:activations:0Rdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2=
;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMulл
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOpReadVariableOpTdcn__attention__parallel__v1_1_dense_layer_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02M
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOp╒
<dcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAddBiasAddEdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul:product:0Sdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2>
<dcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAddў
9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/ReluReluEdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2;
9dcn__attention__parallel__v1_1/dense_layer_1/dense_5/Reluм
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOpSdcn__attention__parallel__v1_1_dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02L
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOp╙
;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMulMatMulGdcn__attention__parallel__v1_1/dense_layer_1/dense_5/Relu:activations:0Rdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2=
;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMulл
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOpTdcn__attention__parallel__v1_1_dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02M
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp╒
<dcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAddBiasAddEdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul:product:0Sdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2>
<dcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAddЮ
,dcn__attention__parallel__v1_1/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,dcn__attention__parallel__v1_1/concat_2/axis╫
'dcn__attention__parallel__v1_1/concat_2ConcatV2=dcn__attention__parallel__v1_1/cross_layer_1/Squeeze:output:0Edcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd:output:05dcn__attention__parallel__v1_1/concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2)
'dcn__attention__parallel__v1_1/concat_2Г
<dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOpReadVariableOpEdcn__attention__parallel__v1_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	▐*
dtype02>
<dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOpТ
-dcn__attention__parallel__v1_1/dense_7/MatMulMatMul0dcn__attention__parallel__v1_1/concat_2:output:0Ddcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2/
-dcn__attention__parallel__v1_1/dense_7/MatMulБ
=dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOpReadVariableOpFdcn__attention__parallel__v1_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOpЭ
.dcn__attention__parallel__v1_1/dense_7/BiasAddBiasAdd7dcn__attention__parallel__v1_1/dense_7/MatMul:product:0Edcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         20
.dcn__attention__parallel__v1_1/dense_7/BiasAdd╞
&dcn__attention__parallel__v1_1/SigmoidSigmoid7dcn__attention__parallel__v1_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2(
&dcn__attention__parallel__v1_1/SigmoidЕ
IdentityIdentity*dcn__attention__parallel__v1_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identity┬
NoOpNoOpI^dcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOpK^dcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOpK^dcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOpK^dcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOp@^dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOpB^dcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOpB^dcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOp@^dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOp>^dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOp=^dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOpL^dcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOpK^dcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOpL^dcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOpK^dcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOpL^dcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOpK^dcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOp=^dcn__attention__parallel__v1_1/embedding_26/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_27/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_28/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_29/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_30/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_31/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_32/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_33/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_34/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_35/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_36/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_37/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_38/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_39/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_40/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_41/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_42/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_43/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_44/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_45/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_46/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_47/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_48/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_49/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_50/embedding_lookup=^dcn__attention__parallel__v1_1/embedding_51/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Ф
Hdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOpHdcn__attention__parallel__v1_1/batch_normalization_1/Cast/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOpJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_1/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOpJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_2/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOpJdcn__attention__parallel__v1_1/batch_normalization_1/Cast_3/ReadVariableOp2В
?dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOp?dcn__attention__parallel__v1_1/cross_layer_1/Mul/ReadVariableOp2Ж
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOpAdcn__attention__parallel__v1_1/cross_layer_1/Mul_1/ReadVariableOp2Ж
Adcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOpAdcn__attention__parallel__v1_1/cross_layer_1/Mul_2/ReadVariableOp2В
?dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOp?dcn__attention__parallel__v1_1/cross_layer_1/add/ReadVariableOp2~
=dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOp=dcn__attention__parallel__v1_1/dense_7/BiasAdd/ReadVariableOp2|
<dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOp<dcn__attention__parallel__v1_1/dense_7/MatMul/ReadVariableOp2Ъ
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOpKdcn__attention__parallel__v1_1/dense_layer_1/dense_4/BiasAdd/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOpJdcn__attention__parallel__v1_1/dense_layer_1/dense_4/MatMul/ReadVariableOp2Ъ
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOpKdcn__attention__parallel__v1_1/dense_layer_1/dense_5/BiasAdd/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOpJdcn__attention__parallel__v1_1/dense_layer_1/dense_5/MatMul/ReadVariableOp2Ъ
Kdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOpKdcn__attention__parallel__v1_1/dense_layer_1/dense_6/BiasAdd/ReadVariableOp2Ш
Jdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOpJdcn__attention__parallel__v1_1/dense_layer_1/dense_6/MatMul/ReadVariableOp2|
<dcn__attention__parallel__v1_1/embedding_26/embedding_lookup<dcn__attention__parallel__v1_1/embedding_26/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_27/embedding_lookup<dcn__attention__parallel__v1_1/embedding_27/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_28/embedding_lookup<dcn__attention__parallel__v1_1/embedding_28/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_29/embedding_lookup<dcn__attention__parallel__v1_1/embedding_29/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_30/embedding_lookup<dcn__attention__parallel__v1_1/embedding_30/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_31/embedding_lookup<dcn__attention__parallel__v1_1/embedding_31/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_32/embedding_lookup<dcn__attention__parallel__v1_1/embedding_32/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_33/embedding_lookup<dcn__attention__parallel__v1_1/embedding_33/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_34/embedding_lookup<dcn__attention__parallel__v1_1/embedding_34/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_35/embedding_lookup<dcn__attention__parallel__v1_1/embedding_35/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_36/embedding_lookup<dcn__attention__parallel__v1_1/embedding_36/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_37/embedding_lookup<dcn__attention__parallel__v1_1/embedding_37/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_38/embedding_lookup<dcn__attention__parallel__v1_1/embedding_38/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_39/embedding_lookup<dcn__attention__parallel__v1_1/embedding_39/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_40/embedding_lookup<dcn__attention__parallel__v1_1/embedding_40/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_41/embedding_lookup<dcn__attention__parallel__v1_1/embedding_41/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_42/embedding_lookup<dcn__attention__parallel__v1_1/embedding_42/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_43/embedding_lookup<dcn__attention__parallel__v1_1/embedding_43/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_44/embedding_lookup<dcn__attention__parallel__v1_1/embedding_44/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_45/embedding_lookup<dcn__attention__parallel__v1_1/embedding_45/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_46/embedding_lookup<dcn__attention__parallel__v1_1/embedding_46/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_47/embedding_lookup<dcn__attention__parallel__v1_1/embedding_47/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_48/embedding_lookup<dcn__attention__parallel__v1_1/embedding_48/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_49/embedding_lookup<dcn__attention__parallel__v1_1/embedding_49/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_50/embedding_lookup<dcn__attention__parallel__v1_1/embedding_50/embedding_lookup2|
<dcn__attention__parallel__v1_1/embedding_51/embedding_lookup<dcn__attention__parallel__v1_1/embedding_51/embedding_lookup:P L
'
_output_shapes
:         '
!
_user_specified_name	input_1
з╒
Ё#
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_65154

inputs5
#embedding_26_embedding_lookup_64788:h6
#embedding_27_embedding_lookup_64798:	 6
#embedding_28_embedding_lookup_64808:	╬6
#embedding_29_embedding_lookup_64818:	┼5
#embedding_30_embedding_lookup_64828:*5
#embedding_31_embedding_lookup_64838:	6
#embedding_32_embedding_lookup_64848:	╨5
#embedding_33_embedding_lookup_64858:C5
#embedding_34_embedding_lookup_64868:6
#embedding_35_embedding_lookup_64878:	╕6
#embedding_36_embedding_lookup_64888:	Т6
#embedding_37_embedding_lookup_64898:	х6
#embedding_38_embedding_lookup_64908:	┌5
#embedding_39_embedding_lookup_64918:6
#embedding_40_embedding_lookup_64928:	╪
6
#embedding_41_embedding_lookup_64938:	╔5
#embedding_42_embedding_lookup_64948:
6
#embedding_43_embedding_lookup_64958:	Г6
#embedding_44_embedding_lookup_64968:	└5
#embedding_45_embedding_lookup_64978:6
#embedding_46_embedding_lookup_64988:	З5
#embedding_47_embedding_lookup_64998:
5
#embedding_48_embedding_lookup_65008:6
#embedding_49_embedding_lookup_65018:	╕5
#embedding_50_embedding_lookup_65028:'6
#embedding_51_embedding_lookup_65038:	╩L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	▌N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	▌A
2batch_normalization_1_cast_readvariableop_resource:	▌C
4batch_normalization_1_cast_1_readvariableop_resource:	▌<
)cross_layer_1_mul_readvariableop_resource:	▌>
+cross_layer_1_mul_1_readvariableop_resource:	▌>
+cross_layer_1_mul_2_readvariableop_resource:	▌<
)cross_layer_1_add_readvariableop_resource:	▌G
4dense_layer_1_dense_4_matmul_readvariableop_resource:	▌ C
5dense_layer_1_dense_4_biasadd_readvariableop_resource: F
4dense_layer_1_dense_5_matmul_readvariableop_resource:  C
5dense_layer_1_dense_5_biasadd_readvariableop_resource: F
4dense_layer_1_dense_6_matmul_readvariableop_resource: C
5dense_layer_1_dense_6_biasadd_readvariableop_resource:9
&dense_7_matmul_readvariableop_resource:	▐5
'dense_7_biasadd_readvariableop_resource:
identityИв%batch_normalization_1/AssignMovingAvgв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв'batch_normalization_1/AssignMovingAvg_1в6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв)batch_normalization_1/Cast/ReadVariableOpв+batch_normalization_1/Cast_1/ReadVariableOpв cross_layer_1/Mul/ReadVariableOpв"cross_layer_1/Mul_1/ReadVariableOpв"cross_layer_1/Mul_2/ReadVariableOpв cross_layer_1/add/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpвdense_7/BiasAdd/ReadVariableOpвdense_7/MatMul/ReadVariableOpв,dense_layer_1/dense_4/BiasAdd/ReadVariableOpв+dense_layer_1/dense_4/MatMul/ReadVariableOpв,dense_layer_1/dense_5/BiasAdd/ReadVariableOpв+dense_layer_1/dense_5/MatMul/ReadVariableOpв,dense_layer_1/dense_6/BiasAdd/ReadVariableOpв+dense_layer_1/dense_6/MatMul/ReadVariableOpвembedding_26/embedding_lookupвembedding_27/embedding_lookupвembedding_28/embedding_lookupвembedding_29/embedding_lookupвembedding_30/embedding_lookupвembedding_31/embedding_lookupвembedding_32/embedding_lookupвembedding_33/embedding_lookupвembedding_34/embedding_lookupвembedding_35/embedding_lookupвembedding_36/embedding_lookupвembedding_37/embedding_lookupвembedding_38/embedding_lookupвembedding_39/embedding_lookupвembedding_40/embedding_lookupвembedding_41/embedding_lookupвembedding_42/embedding_lookupвembedding_43/embedding_lookupвembedding_44/embedding_lookupвembedding_45/embedding_lookupвembedding_46/embedding_lookupвembedding_47/embedding_lookupвembedding_48/embedding_lookupвembedding_49/embedding_lookupвembedding_50/embedding_lookupвembedding_51/embedding_lookup{
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
strided_slice/stack_2ї
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1Г
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2 
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1Г
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2е
strided_slice_2StridedSlicestrided_slice_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2Е
embedding_26/CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_26/Cast╕
embedding_26/embedding_lookupResourceGather#embedding_26_embedding_lookup_64788embedding_26/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_26/embedding_lookup/64788*'
_output_shapes
:         *
dtype02
embedding_26/embedding_lookupЬ
&embedding_26/embedding_lookup/IdentityIdentity&embedding_26/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_26/embedding_lookup/64788*'
_output_shapes
:         2(
&embedding_26/embedding_lookup/Identity├
(embedding_26/embedding_lookup/Identity_1Identity/embedding_26/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_26/embedding_lookup/Identity_1
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stackГ
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1Г
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2е
strided_slice_3StridedSlicestrided_slice_1:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3Е
embedding_27/CastCaststrided_slice_3:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_27/Cast╕
embedding_27/embedding_lookupResourceGather#embedding_27_embedding_lookup_64798embedding_27/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_27/embedding_lookup/64798*'
_output_shapes
:         *
dtype02
embedding_27/embedding_lookupЬ
&embedding_27/embedding_lookup/IdentityIdentity&embedding_27/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_27/embedding_lookup/64798*'
_output_shapes
:         2(
&embedding_27/embedding_lookup/Identity├
(embedding_27/embedding_lookup/Identity_1Identity/embedding_27/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_27/embedding_lookup/Identity_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stackГ
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1Г
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2е
strided_slice_4StridedSlicestrided_slice_1:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4Е
embedding_28/CastCaststrided_slice_4:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_28/Cast╕
embedding_28/embedding_lookupResourceGather#embedding_28_embedding_lookup_64808embedding_28/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_28/embedding_lookup/64808*'
_output_shapes
:         *
dtype02
embedding_28/embedding_lookupЬ
&embedding_28/embedding_lookup/IdentityIdentity&embedding_28/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_28/embedding_lookup/64808*'
_output_shapes
:         2(
&embedding_28/embedding_lookup/Identity├
(embedding_28/embedding_lookup/Identity_1Identity/embedding_28/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_28/embedding_lookup/Identity_1
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stackГ
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1Г
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2е
strided_slice_5StridedSlicestrided_slice_1:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5Е
embedding_29/CastCaststrided_slice_5:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_29/Cast╕
embedding_29/embedding_lookupResourceGather#embedding_29_embedding_lookup_64818embedding_29/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_29/embedding_lookup/64818*'
_output_shapes
:         *
dtype02
embedding_29/embedding_lookupЬ
&embedding_29/embedding_lookup/IdentityIdentity&embedding_29/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_29/embedding_lookup/64818*'
_output_shapes
:         2(
&embedding_29/embedding_lookup/Identity├
(embedding_29/embedding_lookup/Identity_1Identity/embedding_29/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_29/embedding_lookup/Identity_1
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stackГ
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1Г
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2е
strided_slice_6StridedSlicestrided_slice_1:output:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_6Е
embedding_30/CastCaststrided_slice_6:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_30/Cast╕
embedding_30/embedding_lookupResourceGather#embedding_30_embedding_lookup_64828embedding_30/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_30/embedding_lookup/64828*'
_output_shapes
:         *
dtype02
embedding_30/embedding_lookupЬ
&embedding_30/embedding_lookup/IdentityIdentity&embedding_30/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_30/embedding_lookup/64828*'
_output_shapes
:         2(
&embedding_30/embedding_lookup/Identity├
(embedding_30/embedding_lookup/Identity_1Identity/embedding_30/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_30/embedding_lookup/Identity_1
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stackГ
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1Г
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2е
strided_slice_7StridedSlicestrided_slice_1:output:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_7Е
embedding_31/CastCaststrided_slice_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_31/Cast╕
embedding_31/embedding_lookupResourceGather#embedding_31_embedding_lookup_64838embedding_31/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_31/embedding_lookup/64838*'
_output_shapes
:         *
dtype02
embedding_31/embedding_lookupЬ
&embedding_31/embedding_lookup/IdentityIdentity&embedding_31/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_31/embedding_lookup/64838*'
_output_shapes
:         2(
&embedding_31/embedding_lookup/Identity├
(embedding_31/embedding_lookup/Identity_1Identity/embedding_31/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_31/embedding_lookup/Identity_1
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stackГ
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1Г
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2е
strided_slice_8StridedSlicestrided_slice_1:output:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_8Е
embedding_32/CastCaststrided_slice_8:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_32/Cast╕
embedding_32/embedding_lookupResourceGather#embedding_32_embedding_lookup_64848embedding_32/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_32/embedding_lookup/64848*'
_output_shapes
:         *
dtype02
embedding_32/embedding_lookupЬ
&embedding_32/embedding_lookup/IdentityIdentity&embedding_32/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_32/embedding_lookup/64848*'
_output_shapes
:         2(
&embedding_32/embedding_lookup/Identity├
(embedding_32/embedding_lookup/Identity_1Identity/embedding_32/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_32/embedding_lookup/Identity_1
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stackГ
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1Г
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2е
strided_slice_9StridedSlicestrided_slice_1:output:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_9Е
embedding_33/CastCaststrided_slice_9:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_33/Cast╕
embedding_33/embedding_lookupResourceGather#embedding_33_embedding_lookup_64858embedding_33/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_33/embedding_lookup/64858*'
_output_shapes
:         *
dtype02
embedding_33/embedding_lookupЬ
&embedding_33/embedding_lookup/IdentityIdentity&embedding_33/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_33/embedding_lookup/64858*'
_output_shapes
:         2(
&embedding_33/embedding_lookup/Identity├
(embedding_33/embedding_lookup/Identity_1Identity/embedding_33/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_33/embedding_lookup/Identity_1Б
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stackЕ
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_10/stack_1Е
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2к
strided_slice_10StridedSlicestrided_slice_1:output:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_10Ж
embedding_34/CastCaststrided_slice_10:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_34/Cast╕
embedding_34/embedding_lookupResourceGather#embedding_34_embedding_lookup_64868embedding_34/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_34/embedding_lookup/64868*'
_output_shapes
:         *
dtype02
embedding_34/embedding_lookupЬ
&embedding_34/embedding_lookup/IdentityIdentity&embedding_34/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_34/embedding_lookup/64868*'
_output_shapes
:         2(
&embedding_34/embedding_lookup/Identity├
(embedding_34/embedding_lookup/Identity_1Identity/embedding_34/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_34/embedding_lookup/Identity_1Б
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    	   2
strided_slice_11/stackЕ
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_11/stack_1Е
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2к
strided_slice_11StridedSlicestrided_slice_1:output:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_11Ж
embedding_35/CastCaststrided_slice_11:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_35/Cast╕
embedding_35/embedding_lookupResourceGather#embedding_35_embedding_lookup_64878embedding_35/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_35/embedding_lookup/64878*'
_output_shapes
:         *
dtype02
embedding_35/embedding_lookupЬ
&embedding_35/embedding_lookup/IdentityIdentity&embedding_35/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_35/embedding_lookup/64878*'
_output_shapes
:         2(
&embedding_35/embedding_lookup/Identity├
(embedding_35/embedding_lookup/Identity_1Identity/embedding_35/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_35/embedding_lookup/Identity_1Б
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"    
   2
strided_slice_12/stackЕ
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_12/stack_1Е
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2к
strided_slice_12StridedSlicestrided_slice_1:output:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_12Ж
embedding_36/CastCaststrided_slice_12:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_36/Cast╕
embedding_36/embedding_lookupResourceGather#embedding_36_embedding_lookup_64888embedding_36/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_36/embedding_lookup/64888*'
_output_shapes
:         *
dtype02
embedding_36/embedding_lookupЬ
&embedding_36/embedding_lookup/IdentityIdentity&embedding_36/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_36/embedding_lookup/64888*'
_output_shapes
:         2(
&embedding_36/embedding_lookup/Identity├
(embedding_36/embedding_lookup/Identity_1Identity/embedding_36/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_36/embedding_lookup/Identity_1Б
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stackЕ
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_13/stack_1Е
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2к
strided_slice_13StridedSlicestrided_slice_1:output:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_13Ж
embedding_37/CastCaststrided_slice_13:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_37/Cast╕
embedding_37/embedding_lookupResourceGather#embedding_37_embedding_lookup_64898embedding_37/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_37/embedding_lookup/64898*'
_output_shapes
:         *
dtype02
embedding_37/embedding_lookupЬ
&embedding_37/embedding_lookup/IdentityIdentity&embedding_37/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_37/embedding_lookup/64898*'
_output_shapes
:         2(
&embedding_37/embedding_lookup/Identity├
(embedding_37/embedding_lookup/Identity_1Identity/embedding_37/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_37/embedding_lookup/Identity_1Б
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stackЕ
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_14/stack_1Е
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_14/stack_2к
strided_slice_14StridedSlicestrided_slice_1:output:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_14Ж
embedding_38/CastCaststrided_slice_14:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_38/Cast╕
embedding_38/embedding_lookupResourceGather#embedding_38_embedding_lookup_64908embedding_38/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_38/embedding_lookup/64908*'
_output_shapes
:         *
dtype02
embedding_38/embedding_lookupЬ
&embedding_38/embedding_lookup/IdentityIdentity&embedding_38/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_38/embedding_lookup/64908*'
_output_shapes
:         2(
&embedding_38/embedding_lookup/Identity├
(embedding_38/embedding_lookup/Identity_1Identity/embedding_38/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_38/embedding_lookup/Identity_1Б
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stackЕ
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_15/stack_1Е
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_15/stack_2к
strided_slice_15StridedSlicestrided_slice_1:output:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_15Ж
embedding_39/CastCaststrided_slice_15:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_39/Cast╕
embedding_39/embedding_lookupResourceGather#embedding_39_embedding_lookup_64918embedding_39/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_39/embedding_lookup/64918*'
_output_shapes
:         *
dtype02
embedding_39/embedding_lookupЬ
&embedding_39/embedding_lookup/IdentityIdentity&embedding_39/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_39/embedding_lookup/64918*'
_output_shapes
:         2(
&embedding_39/embedding_lookup/Identity├
(embedding_39/embedding_lookup/Identity_1Identity/embedding_39/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_39/embedding_lookup/Identity_1Б
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stackЕ
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_16/stack_1Е
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_16/stack_2к
strided_slice_16StridedSlicestrided_slice_1:output:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_16Ж
embedding_40/CastCaststrided_slice_16:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_40/Cast╕
embedding_40/embedding_lookupResourceGather#embedding_40_embedding_lookup_64928embedding_40/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_40/embedding_lookup/64928*'
_output_shapes
:         *
dtype02
embedding_40/embedding_lookupЬ
&embedding_40/embedding_lookup/IdentityIdentity&embedding_40/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_40/embedding_lookup/64928*'
_output_shapes
:         2(
&embedding_40/embedding_lookup/Identity├
(embedding_40/embedding_lookup/Identity_1Identity/embedding_40/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_40/embedding_lookup/Identity_1Б
strided_slice_17/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stackЕ
strided_slice_17/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_17/stack_1Е
strided_slice_17/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_17/stack_2к
strided_slice_17StridedSlicestrided_slice_1:output:0strided_slice_17/stack:output:0!strided_slice_17/stack_1:output:0!strided_slice_17/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_17Ж
embedding_41/CastCaststrided_slice_17:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_41/Cast╕
embedding_41/embedding_lookupResourceGather#embedding_41_embedding_lookup_64938embedding_41/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_41/embedding_lookup/64938*'
_output_shapes
:         *
dtype02
embedding_41/embedding_lookupЬ
&embedding_41/embedding_lookup/IdentityIdentity&embedding_41/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_41/embedding_lookup/64938*'
_output_shapes
:         2(
&embedding_41/embedding_lookup/Identity├
(embedding_41/embedding_lookup/Identity_1Identity/embedding_41/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_41/embedding_lookup/Identity_1Б
strided_slice_18/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stackЕ
strided_slice_18/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_18/stack_1Е
strided_slice_18/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_18/stack_2к
strided_slice_18StridedSlicestrided_slice_1:output:0strided_slice_18/stack:output:0!strided_slice_18/stack_1:output:0!strided_slice_18/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_18Ж
embedding_42/CastCaststrided_slice_18:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_42/Cast╕
embedding_42/embedding_lookupResourceGather#embedding_42_embedding_lookup_64948embedding_42/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_42/embedding_lookup/64948*'
_output_shapes
:         *
dtype02
embedding_42/embedding_lookupЬ
&embedding_42/embedding_lookup/IdentityIdentity&embedding_42/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_42/embedding_lookup/64948*'
_output_shapes
:         2(
&embedding_42/embedding_lookup/Identity├
(embedding_42/embedding_lookup/Identity_1Identity/embedding_42/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_42/embedding_lookup/Identity_1Б
strided_slice_19/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stackЕ
strided_slice_19/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_19/stack_1Е
strided_slice_19/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_19/stack_2к
strided_slice_19StridedSlicestrided_slice_1:output:0strided_slice_19/stack:output:0!strided_slice_19/stack_1:output:0!strided_slice_19/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_19Ж
embedding_43/CastCaststrided_slice_19:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_43/Cast╕
embedding_43/embedding_lookupResourceGather#embedding_43_embedding_lookup_64958embedding_43/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_43/embedding_lookup/64958*'
_output_shapes
:         *
dtype02
embedding_43/embedding_lookupЬ
&embedding_43/embedding_lookup/IdentityIdentity&embedding_43/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_43/embedding_lookup/64958*'
_output_shapes
:         2(
&embedding_43/embedding_lookup/Identity├
(embedding_43/embedding_lookup/Identity_1Identity/embedding_43/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_43/embedding_lookup/Identity_1Б
strided_slice_20/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stackЕ
strided_slice_20/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_20/stack_1Е
strided_slice_20/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_20/stack_2к
strided_slice_20StridedSlicestrided_slice_1:output:0strided_slice_20/stack:output:0!strided_slice_20/stack_1:output:0!strided_slice_20/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_20Ж
embedding_44/CastCaststrided_slice_20:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_44/Cast╕
embedding_44/embedding_lookupResourceGather#embedding_44_embedding_lookup_64968embedding_44/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_44/embedding_lookup/64968*'
_output_shapes
:         *
dtype02
embedding_44/embedding_lookupЬ
&embedding_44/embedding_lookup/IdentityIdentity&embedding_44/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_44/embedding_lookup/64968*'
_output_shapes
:         2(
&embedding_44/embedding_lookup/Identity├
(embedding_44/embedding_lookup/Identity_1Identity/embedding_44/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_44/embedding_lookup/Identity_1Б
strided_slice_21/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stackЕ
strided_slice_21/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_21/stack_1Е
strided_slice_21/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_21/stack_2к
strided_slice_21StridedSlicestrided_slice_1:output:0strided_slice_21/stack:output:0!strided_slice_21/stack_1:output:0!strided_slice_21/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_21Ж
embedding_45/CastCaststrided_slice_21:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_45/Cast╕
embedding_45/embedding_lookupResourceGather#embedding_45_embedding_lookup_64978embedding_45/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_45/embedding_lookup/64978*'
_output_shapes
:         *
dtype02
embedding_45/embedding_lookupЬ
&embedding_45/embedding_lookup/IdentityIdentity&embedding_45/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_45/embedding_lookup/64978*'
_output_shapes
:         2(
&embedding_45/embedding_lookup/Identity├
(embedding_45/embedding_lookup/Identity_1Identity/embedding_45/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_45/embedding_lookup/Identity_1Б
strided_slice_22/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stackЕ
strided_slice_22/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_22/stack_1Е
strided_slice_22/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_22/stack_2к
strided_slice_22StridedSlicestrided_slice_1:output:0strided_slice_22/stack:output:0!strided_slice_22/stack_1:output:0!strided_slice_22/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_22Ж
embedding_46/CastCaststrided_slice_22:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_46/Cast╕
embedding_46/embedding_lookupResourceGather#embedding_46_embedding_lookup_64988embedding_46/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_46/embedding_lookup/64988*'
_output_shapes
:         *
dtype02
embedding_46/embedding_lookupЬ
&embedding_46/embedding_lookup/IdentityIdentity&embedding_46/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_46/embedding_lookup/64988*'
_output_shapes
:         2(
&embedding_46/embedding_lookup/Identity├
(embedding_46/embedding_lookup/Identity_1Identity/embedding_46/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_46/embedding_lookup/Identity_1Б
strided_slice_23/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stackЕ
strided_slice_23/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_23/stack_1Е
strided_slice_23/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_23/stack_2к
strided_slice_23StridedSlicestrided_slice_1:output:0strided_slice_23/stack:output:0!strided_slice_23/stack_1:output:0!strided_slice_23/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_23Ж
embedding_47/CastCaststrided_slice_23:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_47/Cast╕
embedding_47/embedding_lookupResourceGather#embedding_47_embedding_lookup_64998embedding_47/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_47/embedding_lookup/64998*'
_output_shapes
:         *
dtype02
embedding_47/embedding_lookupЬ
&embedding_47/embedding_lookup/IdentityIdentity&embedding_47/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_47/embedding_lookup/64998*'
_output_shapes
:         2(
&embedding_47/embedding_lookup/Identity├
(embedding_47/embedding_lookup/Identity_1Identity/embedding_47/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_47/embedding_lookup/Identity_1Б
strided_slice_24/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stackЕ
strided_slice_24/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_24/stack_1Е
strided_slice_24/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_24/stack_2к
strided_slice_24StridedSlicestrided_slice_1:output:0strided_slice_24/stack:output:0!strided_slice_24/stack_1:output:0!strided_slice_24/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_24Ж
embedding_48/CastCaststrided_slice_24:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_48/Cast╕
embedding_48/embedding_lookupResourceGather#embedding_48_embedding_lookup_65008embedding_48/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_48/embedding_lookup/65008*'
_output_shapes
:         *
dtype02
embedding_48/embedding_lookupЬ
&embedding_48/embedding_lookup/IdentityIdentity&embedding_48/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_48/embedding_lookup/65008*'
_output_shapes
:         2(
&embedding_48/embedding_lookup/Identity├
(embedding_48/embedding_lookup/Identity_1Identity/embedding_48/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_48/embedding_lookup/Identity_1Б
strided_slice_25/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stackЕ
strided_slice_25/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_25/stack_1Е
strided_slice_25/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_25/stack_2к
strided_slice_25StridedSlicestrided_slice_1:output:0strided_slice_25/stack:output:0!strided_slice_25/stack_1:output:0!strided_slice_25/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_25Ж
embedding_49/CastCaststrided_slice_25:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_49/Cast╕
embedding_49/embedding_lookupResourceGather#embedding_49_embedding_lookup_65018embedding_49/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_49/embedding_lookup/65018*'
_output_shapes
:         *
dtype02
embedding_49/embedding_lookupЬ
&embedding_49/embedding_lookup/IdentityIdentity&embedding_49/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_49/embedding_lookup/65018*'
_output_shapes
:         2(
&embedding_49/embedding_lookup/Identity├
(embedding_49/embedding_lookup/Identity_1Identity/embedding_49/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_49/embedding_lookup/Identity_1Б
strided_slice_26/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stackЕ
strided_slice_26/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_26/stack_1Е
strided_slice_26/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_26/stack_2к
strided_slice_26StridedSlicestrided_slice_1:output:0strided_slice_26/stack:output:0!strided_slice_26/stack_1:output:0!strided_slice_26/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_26Ж
embedding_50/CastCaststrided_slice_26:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_50/Cast╕
embedding_50/embedding_lookupResourceGather#embedding_50_embedding_lookup_65028embedding_50/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_50/embedding_lookup/65028*'
_output_shapes
:         *
dtype02
embedding_50/embedding_lookupЬ
&embedding_50/embedding_lookup/IdentityIdentity&embedding_50/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_50/embedding_lookup/65028*'
_output_shapes
:         2(
&embedding_50/embedding_lookup/Identity├
(embedding_50/embedding_lookup/Identity_1Identity/embedding_50/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_50/embedding_lookup/Identity_1Б
strided_slice_27/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stackЕ
strided_slice_27/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_27/stack_1Е
strided_slice_27/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_27/stack_2к
strided_slice_27StridedSlicestrided_slice_1:output:0strided_slice_27/stack:output:0!strided_slice_27/stack_1:output:0!strided_slice_27/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_27Ж
embedding_51/CastCaststrided_slice_27:output:0*

DstT0*

SrcT0*#
_output_shapes
:         2
embedding_51/Cast╕
embedding_51/embedding_lookupResourceGather#embedding_51_embedding_lookup_65038embedding_51/Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*6
_class,
*(loc:@embedding_51/embedding_lookup/65038*'
_output_shapes
:         *
dtype02
embedding_51/embedding_lookupЬ
&embedding_51/embedding_lookup/IdentityIdentity&embedding_51/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@embedding_51/embedding_lookup/65038*'
_output_shapes
:         2(
&embedding_51/embedding_lookup/Identity├
(embedding_51/embedding_lookup/Identity_1Identity/embedding_51/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2*
(embedding_51/embedding_lookup/Identity_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЬ
concatConcatV21embedding_26/embedding_lookup/Identity_1:output:01embedding_27/embedding_lookup/Identity_1:output:01embedding_28/embedding_lookup/Identity_1:output:01embedding_29/embedding_lookup/Identity_1:output:01embedding_30/embedding_lookup/Identity_1:output:01embedding_31/embedding_lookup/Identity_1:output:01embedding_32/embedding_lookup/Identity_1:output:01embedding_33/embedding_lookup/Identity_1:output:01embedding_34/embedding_lookup/Identity_1:output:01embedding_35/embedding_lookup/Identity_1:output:01embedding_36/embedding_lookup/Identity_1:output:01embedding_37/embedding_lookup/Identity_1:output:01embedding_38/embedding_lookup/Identity_1:output:01embedding_39/embedding_lookup/Identity_1:output:01embedding_40/embedding_lookup/Identity_1:output:01embedding_41/embedding_lookup/Identity_1:output:01embedding_42/embedding_lookup/Identity_1:output:01embedding_43/embedding_lookup/Identity_1:output:01embedding_44/embedding_lookup/Identity_1:output:01embedding_45/embedding_lookup/Identity_1:output:01embedding_46/embedding_lookup/Identity_1:output:01embedding_47/embedding_lookup/Identity_1:output:01embedding_48/embedding_lookup/Identity_1:output:01embedding_49/embedding_lookup/Identity_1:output:01embedding_50/embedding_lookup/Identity_1:output:01embedding_51/embedding_lookup/Identity_1:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:         ╨2
concat`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axisЭ
concat_1ConcatV2strided_slice:output:0concat:output:0concat_1/axis:output:0*
N*
T0*(
_output_shapes
:         ▌2

concat_1╢
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_1/moments/mean/reduction_indices▌
"batch_normalization_1/moments/meanMeanconcat_1:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2$
"batch_normalization_1/moments/mean┐
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	▌2,
*batch_normalization_1/moments/StopGradientЄ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconcat_1:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ▌21
/batch_normalization_1/moments/SquaredDifference╛
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_1/moments/variance/reduction_indicesЛ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	▌*
	keep_dims(2(
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:▌*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Я
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decayч
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:▌*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpё
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:▌2+
)batch_normalization_1/AssignMovingAvg/subш
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:▌2+
)batch_normalization_1/AssignMovingAvg/mulн
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvgг
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decayэ
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:▌*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp∙
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:▌2-
+batch_normalization_1/AssignMovingAvg_1/subЁ
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:▌2-
+batch_normalization_1/AssignMovingAvg_1/mul╖
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1╞
)batch_normalization_1/Cast/ReadVariableOpReadVariableOp2batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:▌*
dtype02+
)batch_normalization_1/Cast/ReadVariableOp╠
+batch_normalization_1/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:▌*
dtype02-
+batch_normalization_1/Cast_1/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y█
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/addж
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:▌2'
%batch_normalization_1/batchnorm/Rsqrt╫
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:03batch_normalization_1/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/mul─
%batch_normalization_1/batchnorm/mul_1Mulconcat_1:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ▌2'
%batch_normalization_1/batchnorm/mul_1╘
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:▌2'
%batch_normalization_1/batchnorm/mul_2╒
#batch_normalization_1/batchnorm/subSub1batch_normalization_1/Cast/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:▌2%
#batch_normalization_1/batchnorm/sub▐
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ▌2'
%batch_normalization_1/batchnorm/add_1~
cross_layer_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
cross_layer_1/ExpandDims/dim╦
cross_layer_1/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0%cross_layer_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/ExpandDimsп
 cross_layer_1/Mul/ReadVariableOpReadVariableOp)cross_layer_1_mul_readvariableop_resource*
_output_shapes
:	▌*
dtype02"
 cross_layer_1/Mul/ReadVariableOp▒
cross_layer_1/MulMul!cross_layer_1/ExpandDims:output:0(cross_layer_1/Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul╡
"cross_layer_1/Mul_1/ReadVariableOpReadVariableOp+cross_layer_1_mul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02$
"cross_layer_1/Mul_1/ReadVariableOp╖
cross_layer_1/Mul_1Mul!cross_layer_1/ExpandDims:output:0*cross_layer_1/Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_1╡
"cross_layer_1/Mul_2/ReadVariableOpReadVariableOp+cross_layer_1_mul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02$
"cross_layer_1/Mul_2/ReadVariableOp╖
cross_layer_1/Mul_2Mul!cross_layer_1/ExpandDims:output:0*cross_layer_1/Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_2▓
cross_layer_1/MatMulBatchMatMulV2cross_layer_1/Mul:z:0cross_layer_1/Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
cross_layer_1/MatMulД
cross_layer_1/ExpExpcross_layer_1/MatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
cross_layer_1/ExpМ
#cross_layer_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#cross_layer_1/Sum/reduction_indices║
cross_layer_1/SumSumcross_layer_1/Exp:y:0,cross_layer_1/Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
cross_layer_1/Sumд
cross_layer_1/truedivRealDivcross_layer_1/Exp:y:0cross_layer_1/Sum:output:0*
T0*-
_output_shapes
:         ▌▌2
cross_layer_1/truediv╢
cross_layer_1/MatMul_1BatchMatMulV2cross_layer_1/truediv:z:0!cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/MatMul_1в
cross_layer_1/Mul_3Mulcross_layer_1/Mul_2:z:0cross_layer_1/MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/Mul_3п
 cross_layer_1/add/ReadVariableOpReadVariableOp)cross_layer_1_add_readvariableop_resource*
_output_shapes
:	▌*
dtype02"
 cross_layer_1/add/ReadVariableOpй
cross_layer_1/addAddV2cross_layer_1/Mul_3:z:0(cross_layer_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/addд
cross_layer_1/add_1AddV2cross_layer_1/add:z:0!cross_layer_1/ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
cross_layer_1/add_1Ь
cross_layer_1/SqueezeSqueezecross_layer_1/add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2
cross_layer_1/Squeeze╨
+dense_layer_1/dense_4/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02-
+dense_layer_1/dense_4/MatMul/ReadVariableOp╪
dense_layer_1/dense_4/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:03dense_layer_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/MatMul╬
,dense_layer_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,dense_layer_1/dense_4/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_4/BiasAddBiasAdd&dense_layer_1/dense_4/MatMul:product:04dense_layer_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/BiasAddЪ
dense_layer_1/dense_4/ReluRelu&dense_layer_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_4/Relu╧
+dense_layer_1/dense_5/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+dense_layer_1/dense_5/MatMul/ReadVariableOp╫
dense_layer_1/dense_5/MatMulMatMul(dense_layer_1/dense_4/Relu:activations:03dense_layer_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/MatMul╬
,dense_layer_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,dense_layer_1/dense_5/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_5/BiasAddBiasAdd&dense_layer_1/dense_5/MatMul:product:04dense_layer_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/BiasAddЪ
dense_layer_1/dense_5/ReluRelu&dense_layer_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_layer_1/dense_5/Relu╧
+dense_layer_1/dense_6/MatMul/ReadVariableOpReadVariableOp4dense_layer_1_dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+dense_layer_1/dense_6/MatMul/ReadVariableOp╫
dense_layer_1/dense_6/MatMulMatMul(dense_layer_1/dense_5/Relu:activations:03dense_layer_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_layer_1/dense_6/MatMul╬
,dense_layer_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp5dense_layer_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp┘
dense_layer_1/dense_6/BiasAddBiasAdd&dense_layer_1/dense_6/MatMul:product:04dense_layer_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_layer_1/dense_6/BiasAdd`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis╝
concat_2ConcatV2cross_layer_1/Squeeze:output:0&dense_layer_1/dense_6/BiasAdd:output:0concat_2/axis:output:0*
N*
T0*(
_output_shapes
:         ▐2

concat_2ж
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	▐*
dtype02
dense_7/MatMul/ReadVariableOpЦ
dense_7/MatMulMatMulconcat_2:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/MatMulд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpб
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/BiasAddi
SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidХ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOp+cross_layer_1_mul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulУ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOp)cross_layer_1_mul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulХ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOp+cross_layer_1_mul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulС
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOp)cross_layer_1_add_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         2

Identityх
NoOpNoOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_1/Cast/ReadVariableOp,^batch_normalization_1/Cast_1/ReadVariableOp!^cross_layer_1/Mul/ReadVariableOp#^cross_layer_1/Mul_1/ReadVariableOp#^cross_layer_1/Mul_2/ReadVariableOp!^cross_layer_1/add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp-^dense_layer_1/dense_4/BiasAdd/ReadVariableOp,^dense_layer_1/dense_4/MatMul/ReadVariableOp-^dense_layer_1/dense_5/BiasAdd/ReadVariableOp,^dense_layer_1/dense_5/MatMul/ReadVariableOp-^dense_layer_1/dense_6/BiasAdd/ReadVariableOp,^dense_layer_1/dense_6/MatMul/ReadVariableOp^embedding_26/embedding_lookup^embedding_27/embedding_lookup^embedding_28/embedding_lookup^embedding_29/embedding_lookup^embedding_30/embedding_lookup^embedding_31/embedding_lookup^embedding_32/embedding_lookup^embedding_33/embedding_lookup^embedding_34/embedding_lookup^embedding_35/embedding_lookup^embedding_36/embedding_lookup^embedding_37/embedding_lookup^embedding_38/embedding_lookup^embedding_39/embedding_lookup^embedding_40/embedding_lookup^embedding_41/embedding_lookup^embedding_42/embedding_lookup^embedding_43/embedding_lookup^embedding_44/embedding_lookup^embedding_45/embedding_lookup^embedding_46/embedding_lookup^embedding_47/embedding_lookup^embedding_48/embedding_lookup^embedding_49/embedding_lookup^embedding_50/embedding_lookup^embedding_51/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         ': : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_1/Cast/ReadVariableOp)batch_normalization_1/Cast/ReadVariableOp2Z
+batch_normalization_1/Cast_1/ReadVariableOp+batch_normalization_1/Cast_1/ReadVariableOp2D
 cross_layer_1/Mul/ReadVariableOp cross_layer_1/Mul/ReadVariableOp2H
"cross_layer_1/Mul_1/ReadVariableOp"cross_layer_1/Mul_1/ReadVariableOp2H
"cross_layer_1/Mul_2/ReadVariableOp"cross_layer_1/Mul_2/ReadVariableOp2D
 cross_layer_1/add/ReadVariableOp cross_layer_1/add/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2\
,dense_layer_1/dense_4/BiasAdd/ReadVariableOp,dense_layer_1/dense_4/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_4/MatMul/ReadVariableOp+dense_layer_1/dense_4/MatMul/ReadVariableOp2\
,dense_layer_1/dense_5/BiasAdd/ReadVariableOp,dense_layer_1/dense_5/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_5/MatMul/ReadVariableOp+dense_layer_1/dense_5/MatMul/ReadVariableOp2\
,dense_layer_1/dense_6/BiasAdd/ReadVariableOp,dense_layer_1/dense_6/BiasAdd/ReadVariableOp2Z
+dense_layer_1/dense_6/MatMul/ReadVariableOp+dense_layer_1/dense_6/MatMul/ReadVariableOp2>
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
embedding_51/embedding_lookupembedding_51/embedding_lookup:O K
'
_output_shapes
:         '
 
_user_specified_nameinputs
╦

г
G__inference_embedding_45_layer_call_and_return_conditional_losses_62523

inputs(
embedding_lookup_62517:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62517Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62517*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62517*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╩
А
,__inference_embedding_45_layer_call_fn_66049

inputs
unknown:
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_45_layer_call_and_return_conditional_losses_625232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_27_layer_call_fn_65743

inputs
unknown:	 
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_27_layer_call_and_return_conditional_losses_621992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
█W
╓
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65569

inputs.
mul_readvariableop_resource:	▌0
mul_1_readvariableop_resource:	▌0
mul_2_readvariableop_resource:	▌.
add_readvariableop_resource:	▌
identityИвMul/ReadVariableOpвMul_1/ReadVariableOpвMul_2/ReadVariableOpвadd/ReadVariableOpвQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpвRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:         ▌2

ExpandDimsЕ
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul/ReadVariableOpy
MulMulExpandDims:output:0Mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
MulЛ
Mul_1/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_1/ReadVariableOp
Mul_1MulExpandDims:output:0Mul_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_1Л
Mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02
Mul_2/ReadVariableOp
Mul_2MulExpandDims:output:0Mul_2/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
Mul_2z
MatMulBatchMatMulV2Mul:z:0	Mul_1:z:0*
T0*-
_output_shapes
:         ▌▌*
adj_y(2
MatMulZ
ExpExpMatMul:output:0*
T0*-
_output_shapes
:         ▌▌2
Expp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesВ
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*,
_output_shapes
:         ▌*
	keep_dims(2
Suml
truedivRealDivExp:y:0Sum:output:0*
T0*-
_output_shapes
:         ▌▌2	
truediv~
MatMul_1BatchMatMulV2truediv:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2

MatMul_1j
Mul_3Mul	Mul_2:z:0MatMul_1:output:0*
T0*,
_output_shapes
:         ▌2
Mul_3Е
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02
add/ReadVariableOpq
addAddV2	Mul_3:z:0add/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ▌2
addl
add_1AddV2add:z:0ExpandDims:output:0*
T0*,
_output_shapes
:         ▌2
add_1r
SqueezeSqueeze	add_1:z:0*
T0*(
_output_shapes
:         ▌*
squeeze_dims
2	
SqueezeЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpReadVariableOpmul_1_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/mulЕ
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/mulЗ
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:	▌*
dtype02T
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpЪ
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SquareSquareZdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2E
Cdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square┘
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const┬
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/SumSumGdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square:y:0Kdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum═
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82D
Bdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x─
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulMulKdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mul/x:output:0Idcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2B
@dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/mulГ
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:	▌*
dtype02S
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpЧ
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SquareSquareYdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	▌2D
Bdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square╫
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const╛
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/SumSumFdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square:y:0Jdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum╦
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82C
Adcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x└
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mulMulJdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mul/x:output:0Hdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/mull
IdentityIdentitySqueeze:output:0^NoOp*
T0*(
_output_shapes
:         ▌2

Identity∙
NoOpNoOp^Mul/ReadVariableOp^Mul_1/ReadVariableOp^Mul_2/ReadVariableOp^add/ReadVariableOpR^dcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpS^dcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ▌: : : : 2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul_1/ReadVariableOpMul_1/ReadVariableOp2,
Mul_2/ReadVariableOpMul_2/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2ж
Qdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOpQdcn__attention__parallel__v1_1/cross_layer_1/b0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wk0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wq0/Regularizer/Square/ReadVariableOp2и
Rdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOpRdcn__attention__parallel__v1_1/cross_layer_1/wv0/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_39_layer_call_and_return_conditional_losses_65940

inputs(
embedding_lookup_65934:
identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_65934Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/65934*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/65934*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦

г
G__inference_embedding_47_layer_call_and_return_conditional_losses_66076

inputs(
embedding_lookup_66070:

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_66070Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/66070*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/66070*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
Ё
■
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65436

inputs9
&dense_4_matmul_readvariableop_resource:	▌ 5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource:  5
'dense_5_biasadd_readvariableop_resource: 8
&dense_6_matmul_readvariableop_resource: 5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpвdense_6/BiasAdd/ReadVariableOpвdense_6/MatMul/ReadVariableOpж
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	▌ *
dtype02
dense_4/MatMul/ReadVariableOpЛ
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_5/BiasAddp
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_5/Reluе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/MatMulд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/BiasAdds
IdentityIdentitydense_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityС
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ▌: : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:P L
(
_output_shapes
:         ▌
 
_user_specified_nameinputs
╦

г
G__inference_embedding_42_layer_call_and_return_conditional_losses_62469

inputs(
embedding_lookup_62463:

identityИвembedding_lookupY
CastCastinputs*

DstT0*

SrcT0*#
_output_shapes
:         2
Castў
embedding_lookupResourceGatherembedding_lookup_62463Cast:y:0",/job:localhost/replica:0/task:0/device:GPU:0*
Tindices0*)
_class
loc:@embedding_lookup/62463*'
_output_shapes
:         *
dtype02
embedding_lookupш
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*)
_class
loc:@embedding_lookup/62463*'
_output_shapes
:         2
embedding_lookup/IdentityЬ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:         2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:         2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
╦
Б
,__inference_embedding_35_layer_call_fn_65879

inputs
unknown:	╕
identityИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_embedding_35_layer_call_and_return_conditional_losses_623432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:         : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
;
input_10
serving_default_input_1:0         '<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:А╡
щ
dense_feature_columns
sparse_feature_columns
embed_layers
bn
dense_layer
cross_layer
output_layer
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
+Ч&call_and_return_all_conditional_losses
Ш_default_save_signature
Щ__call__"
_tf_keras_model
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
ц
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
.19
/20
021
122
223
324
425"
trackable_list_wrapper
В
5embed_0
6embed_1
7embed_2
8embed_3
9embed_4
:embed_5
;embed_6
<embed_7
=embed_8
>embed_9
?embed_10
@embed_11
Aembed_12
Bembed_13
Cembed_14
Dembed_15
Eembed_16
Fembed_17
Gembed_18
Hembed_19
Iembed_20
Jembed_21
Kembed_22
Lembed_23
Membed_24
Nembed_25"
trackable_dict_wrapper
ь
Oaxis
	Pgamma
Qbeta
Rmoving_mean
Smoving_variance
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"
_tf_keras_layer
╦
Xhidden_layer
Youtput_layer
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"
_tf_keras_layer
Є
^wq0
_wq
`wk0
awk
bwv0
cwv
db0
e
cross_bias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"
_tf_keras_layer
╜

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+а&call_and_return_all_conditional_losses
б__call__"
_tf_keras_layer
Э
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratePm╟Qm╚^m╔`m╩bm╦dm╠jm═km╬um╧vm╨wm╤xm╥ym╙zm╘{m╒|m╓}m╫~m╪m┘	Аm┌	Бm█	Вm▄	Гm▌	Дm▐	Еm▀	Жmр	Зmс	Иmт	Йmу	Кmф	Лmх	Мmц	Нmч	Оmш	Пmщ	Рmъ	Сmы	Тmь	Уmэ	ФmюPvяQvЁ^vё`vЄbvєdvЇjvїkvЎuvўvv°wv∙xv·yv√zv№{v¤|v■}v ~vАvБ	АvВ	БvГ	ВvД	ГvЕ	ДvЖ	ЕvЗ	ЖvИ	ЗvЙ	ИvК	ЙvЛ	КvМ	ЛvН	МvО	НvП	ОvР	ПvС	РvТ	СvУ	ТvФ	УvХ	ФvЦ"
	optimizer
@
в0
г1
д2
е3"
trackable_list_wrapper
√
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
P26
Q27
R28
S29
П30
Р31
С32
Т33
У34
Ф35
^36
`37
b38
d39
j40
k41"
trackable_list_wrapper
ы
u0
v1
w2
x3
y4
z5
{6
|7
}8
~9
10
А11
Б12
В13
Г14
Д15
Е16
Ж17
З18
И19
Й20
К21
Л22
М23
Н24
О25
P26
Q27
П28
Р29
С30
Т31
У32
Ф33
^34
`35
b36
d37
j38
k39"
trackable_list_wrapper
╙
Хlayers
Цnon_trainable_variables
Чlayer_metrics
	regularization_losses
Шmetrics

	variables
 Щlayer_regularization_losses
trainable_variables
Щ__call__
Ш_default_save_signature
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
-
жserving_default"
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
╗
u
embeddings
Ъregularization_losses
Ы	variables
Ьtrainable_variables
Э	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
╗
v
embeddings
Юregularization_losses
Я	variables
аtrainable_variables
б	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_layer
╝
Б
embeddings
вregularization_losses
г	variables
дtrainable_variables
е	keras_api
+л&call_and_return_all_conditional_losses
м__call__"
_tf_keras_layer
╝
И
embeddings
жregularization_losses
з	variables
иtrainable_variables
й	keras_api
+н&call_and_return_all_conditional_losses
о__call__"
_tf_keras_layer
╝
Й
embeddings
кregularization_losses
л	variables
мtrainable_variables
н	keras_api
+п&call_and_return_all_conditional_losses
░__call__"
_tf_keras_layer
╝
К
embeddings
оregularization_losses
п	variables
░trainable_variables
▒	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"
_tf_keras_layer
╝
Л
embeddings
▓regularization_losses
│	variables
┤trainable_variables
╡	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"
_tf_keras_layer
╝
М
embeddings
╢regularization_losses
╖	variables
╕trainable_variables
╣	keras_api
+╡&call_and_return_all_conditional_losses
╢__call__"
_tf_keras_layer
╝
Н
embeddings
║regularization_losses
╗	variables
╝trainable_variables
╜	keras_api
+╖&call_and_return_all_conditional_losses
╕__call__"
_tf_keras_layer
╝
О
embeddings
╛regularization_losses
┐	variables
└trainable_variables
┴	keras_api
+╣&call_and_return_all_conditional_losses
║__call__"
_tf_keras_layer
╗
w
embeddings
┬regularization_losses
├	variables
─trainable_variables
┼	keras_api
+╗&call_and_return_all_conditional_losses
╝__call__"
_tf_keras_layer
╗
x
embeddings
╞regularization_losses
╟	variables
╚trainable_variables
╔	keras_api
+╜&call_and_return_all_conditional_losses
╛__call__"
_tf_keras_layer
╗
y
embeddings
╩regularization_losses
╦	variables
╠trainable_variables
═	keras_api
+┐&call_and_return_all_conditional_losses
└__call__"
_tf_keras_layer
╗
z
embeddings
╬regularization_losses
╧	variables
╨trainable_variables
╤	keras_api
+┴&call_and_return_all_conditional_losses
┬__call__"
_tf_keras_layer
╗
{
embeddings
╥regularization_losses
╙	variables
╘trainable_variables
╒	keras_api
+├&call_and_return_all_conditional_losses
─__call__"
_tf_keras_layer
╗
|
embeddings
╓regularization_losses
╫	variables
╪trainable_variables
┘	keras_api
+┼&call_and_return_all_conditional_losses
╞__call__"
_tf_keras_layer
╗
}
embeddings
┌regularization_losses
█	variables
▄trainable_variables
▌	keras_api
+╟&call_and_return_all_conditional_losses
╚__call__"
_tf_keras_layer
╗
~
embeddings
▐regularization_losses
▀	variables
рtrainable_variables
с	keras_api
+╔&call_and_return_all_conditional_losses
╩__call__"
_tf_keras_layer
╗

embeddings
тregularization_losses
у	variables
фtrainable_variables
х	keras_api
+╦&call_and_return_all_conditional_losses
╠__call__"
_tf_keras_layer
╝
А
embeddings
цregularization_losses
ч	variables
шtrainable_variables
щ	keras_api
+═&call_and_return_all_conditional_losses
╬__call__"
_tf_keras_layer
╝
В
embeddings
ъregularization_losses
ы	variables
ьtrainable_variables
э	keras_api
+╧&call_and_return_all_conditional_losses
╨__call__"
_tf_keras_layer
╝
Г
embeddings
юregularization_losses
я	variables
Ёtrainable_variables
ё	keras_api
+╤&call_and_return_all_conditional_losses
╥__call__"
_tf_keras_layer
╝
Д
embeddings
Єregularization_losses
є	variables
Їtrainable_variables
ї	keras_api
+╙&call_and_return_all_conditional_losses
╘__call__"
_tf_keras_layer
╝
Е
embeddings
Ўregularization_losses
ў	variables
°trainable_variables
∙	keras_api
+╒&call_and_return_all_conditional_losses
╓__call__"
_tf_keras_layer
╝
Ж
embeddings
·regularization_losses
√	variables
№trainable_variables
¤	keras_api
+╫&call_and_return_all_conditional_losses
╪__call__"
_tf_keras_layer
╝
З
embeddings
■regularization_losses
 	variables
Аtrainable_variables
Б	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"
_tf_keras_layer
 "
trackable_list_wrapper
I:G▌2:dcn__attention__parallel__v1_1/batch_normalization_1/gamma
H:F▌29dcn__attention__parallel__v1_1/batch_normalization_1/beta
Q:O▌ (2@dcn__attention__parallel__v1_1/batch_normalization_1/moving_mean
U:S▌ (2Ddcn__attention__parallel__v1_1/batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
╡
Вlayers
Гnon_trainable_variables
Дlayer_metrics
Tregularization_losses
Еmetrics
U	variables
 Жlayer_regularization_losses
Vtrainable_variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
0
З0
И1"
trackable_list_wrapper
├
Уkernel
	Фbias
Йregularization_losses
К	variables
Лtrainable_variables
М	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"
_tf_keras_layer
 "
trackable_list_wrapper
P
П0
Р1
С2
Т3
У4
Ф5"
trackable_list_wrapper
P
П0
Р1
С2
Т3
У4
Ф5"
trackable_list_wrapper
╡
Нlayers
Оnon_trainable_variables
Пlayer_metrics
Zregularization_losses
Рmetrics
[	variables
 Сlayer_regularization_losses
\trainable_variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
C:A	▌20dcn__attention__parallel__v1_1/cross_layer_1/wq0
'
^0"
trackable_list_wrapper
C:A	▌20dcn__attention__parallel__v1_1/cross_layer_1/wk0
'
`0"
trackable_list_wrapper
C:A	▌20dcn__attention__parallel__v1_1/cross_layer_1/wv0
'
b0"
trackable_list_wrapper
B:@	▌2/dcn__attention__parallel__v1_1/cross_layer_1/b0
'
d0"
trackable_list_wrapper
@
в0
г1
д2
е3"
trackable_list_wrapper
<
^0
`1
b2
d3"
trackable_list_wrapper
<
^0
`1
b2
d3"
trackable_list_wrapper
╡
Тlayers
Уnon_trainable_variables
Фlayer_metrics
fregularization_losses
Хmetrics
g	variables
 Цlayer_regularization_losses
htrainable_variables
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
@:>	▐2-dcn__attention__parallel__v1_1/dense_7/kernel
9:72+dcn__attention__parallel__v1_1/dense_7/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
╡
Чlayers
Шnon_trainable_variables
Щlayer_metrics
lregularization_losses
Ъmetrics
m	variables
 Ыlayer_regularization_losses
ntrainable_variables
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
H:Fh26dcn__attention__parallel__v1_1/embedding_26/embeddings
I:G	 26dcn__attention__parallel__v1_1/embedding_27/embeddings
I:G	Т26dcn__attention__parallel__v1_1/embedding_36/embeddings
I:G	х26dcn__attention__parallel__v1_1/embedding_37/embeddings
I:G	┌26dcn__attention__parallel__v1_1/embedding_38/embeddings
H:F26dcn__attention__parallel__v1_1/embedding_39/embeddings
I:G	╪
26dcn__attention__parallel__v1_1/embedding_40/embeddings
I:G	╔26dcn__attention__parallel__v1_1/embedding_41/embeddings
H:F
26dcn__attention__parallel__v1_1/embedding_42/embeddings
I:G	Г26dcn__attention__parallel__v1_1/embedding_43/embeddings
I:G	└26dcn__attention__parallel__v1_1/embedding_44/embeddings
H:F26dcn__attention__parallel__v1_1/embedding_45/embeddings
I:G	╬26dcn__attention__parallel__v1_1/embedding_28/embeddings
I:G	З26dcn__attention__parallel__v1_1/embedding_46/embeddings
H:F
26dcn__attention__parallel__v1_1/embedding_47/embeddings
H:F26dcn__attention__parallel__v1_1/embedding_48/embeddings
I:G	╕26dcn__attention__parallel__v1_1/embedding_49/embeddings
H:F'26dcn__attention__parallel__v1_1/embedding_50/embeddings
I:G	╩26dcn__attention__parallel__v1_1/embedding_51/embeddings
I:G	┼26dcn__attention__parallel__v1_1/embedding_29/embeddings
H:F*26dcn__attention__parallel__v1_1/embedding_30/embeddings
H:F	26dcn__attention__parallel__v1_1/embedding_31/embeddings
I:G	╨26dcn__attention__parallel__v1_1/embedding_32/embeddings
H:FC26dcn__attention__parallel__v1_1/embedding_33/embeddings
H:F26dcn__attention__parallel__v1_1/embedding_34/embeddings
I:G	╕26dcn__attention__parallel__v1_1/embedding_35/embeddings
N:L	▌ 2;dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel
G:E 29dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias
M:K  2;dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel
G:E 29dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias
M:K 2;dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel
G:E29dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias
Ж
50
61
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
712
I13
J14
K15
L16
M17
N18
819
920
:21
;22
<23
=24
>25
26
27
28
29"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_dict_wrapper
8
Ь0
Э1
Ю2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
u0"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
╕
Яlayers
аnon_trainable_variables
бlayer_metrics
Ъregularization_losses
вmetrics
Ы	variables
 гlayer_regularization_losses
Ьtrainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
v0"
trackable_list_wrapper
'
v0"
trackable_list_wrapper
╕
дlayers
еnon_trainable_variables
жlayer_metrics
Юregularization_losses
зmetrics
Я	variables
 иlayer_regularization_losses
аtrainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Б0"
trackable_list_wrapper
(
Б0"
trackable_list_wrapper
╕
йlayers
кnon_trainable_variables
лlayer_metrics
вregularization_losses
мmetrics
г	variables
 нlayer_regularization_losses
дtrainable_variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
И0"
trackable_list_wrapper
(
И0"
trackable_list_wrapper
╕
оlayers
пnon_trainable_variables
░layer_metrics
жregularization_losses
▒metrics
з	variables
 ▓layer_regularization_losses
иtrainable_variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Й0"
trackable_list_wrapper
(
Й0"
trackable_list_wrapper
╕
│layers
┤non_trainable_variables
╡layer_metrics
кregularization_losses
╢metrics
л	variables
 ╖layer_regularization_losses
мtrainable_variables
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
К0"
trackable_list_wrapper
(
К0"
trackable_list_wrapper
╕
╕layers
╣non_trainable_variables
║layer_metrics
оregularization_losses
╗metrics
п	variables
 ╝layer_regularization_losses
░trainable_variables
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Л0"
trackable_list_wrapper
(
Л0"
trackable_list_wrapper
╕
╜layers
╛non_trainable_variables
┐layer_metrics
▓regularization_losses
└metrics
│	variables
 ┴layer_regularization_losses
┤trainable_variables
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
М0"
trackable_list_wrapper
(
М0"
trackable_list_wrapper
╕
┬layers
├non_trainable_variables
─layer_metrics
╢regularization_losses
┼metrics
╖	variables
 ╞layer_regularization_losses
╕trainable_variables
╢__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
(
Н0"
trackable_list_wrapper
╕
╟layers
╚non_trainable_variables
╔layer_metrics
║regularization_losses
╩metrics
╗	variables
 ╦layer_regularization_losses
╝trainable_variables
╕__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
О0"
trackable_list_wrapper
(
О0"
trackable_list_wrapper
╕
╠layers
═non_trainable_variables
╬layer_metrics
╛regularization_losses
╧metrics
┐	variables
 ╨layer_regularization_losses
└trainable_variables
║__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
w0"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
╕
╤layers
╥non_trainable_variables
╙layer_metrics
┬regularization_losses
╘metrics
├	variables
 ╒layer_regularization_losses
─trainable_variables
╝__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
x0"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
╕
╓layers
╫non_trainable_variables
╪layer_metrics
╞regularization_losses
┘metrics
╟	variables
 ┌layer_regularization_losses
╚trainable_variables
╛__call__
+╜&call_and_return_all_conditional_losses
'╜"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
y0"
trackable_list_wrapper
'
y0"
trackable_list_wrapper
╕
█layers
▄non_trainable_variables
▌layer_metrics
╩regularization_losses
▐metrics
╦	variables
 ▀layer_regularization_losses
╠trainable_variables
└__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
z0"
trackable_list_wrapper
'
z0"
trackable_list_wrapper
╕
рlayers
сnon_trainable_variables
тlayer_metrics
╬regularization_losses
уmetrics
╧	variables
 фlayer_regularization_losses
╨trainable_variables
┬__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
╕
хlayers
цnon_trainable_variables
чlayer_metrics
╥regularization_losses
шmetrics
╙	variables
 щlayer_regularization_losses
╘trainable_variables
─__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
|0"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
╕
ъlayers
ыnon_trainable_variables
ьlayer_metrics
╓regularization_losses
эmetrics
╫	variables
 юlayer_regularization_losses
╪trainable_variables
╞__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
}0"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
╕
яlayers
Ёnon_trainable_variables
ёlayer_metrics
┌regularization_losses
Єmetrics
█	variables
 єlayer_regularization_losses
▄trainable_variables
╚__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
╕
Їlayers
їnon_trainable_variables
Ўlayer_metrics
▐regularization_losses
ўmetrics
▀	variables
 °layer_regularization_losses
рtrainable_variables
╩__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
╕
∙layers
·non_trainable_variables
√layer_metrics
тregularization_losses
№metrics
у	variables
 ¤layer_regularization_losses
фtrainable_variables
╠__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
А0"
trackable_list_wrapper
(
А0"
trackable_list_wrapper
╕
■layers
 non_trainable_variables
Аlayer_metrics
цregularization_losses
Бmetrics
ч	variables
 Вlayer_regularization_losses
шtrainable_variables
╬__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
В0"
trackable_list_wrapper
(
В0"
trackable_list_wrapper
╕
Гlayers
Дnon_trainable_variables
Еlayer_metrics
ъregularization_losses
Жmetrics
ы	variables
 Зlayer_regularization_losses
ьtrainable_variables
╨__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Г0"
trackable_list_wrapper
(
Г0"
trackable_list_wrapper
╕
Иlayers
Йnon_trainable_variables
Кlayer_metrics
юregularization_losses
Лmetrics
я	variables
 Мlayer_regularization_losses
Ёtrainable_variables
╥__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Д0"
trackable_list_wrapper
(
Д0"
trackable_list_wrapper
╕
Нlayers
Оnon_trainable_variables
Пlayer_metrics
Єregularization_losses
Рmetrics
є	variables
 Сlayer_regularization_losses
Їtrainable_variables
╘__call__
+╙&call_and_return_all_conditional_losses
'╙"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Е0"
trackable_list_wrapper
(
Е0"
trackable_list_wrapper
╕
Тlayers
Уnon_trainable_variables
Фlayer_metrics
Ўregularization_losses
Хmetrics
ў	variables
 Цlayer_regularization_losses
°trainable_variables
╓__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
Ж0"
trackable_list_wrapper
(
Ж0"
trackable_list_wrapper
╕
Чlayers
Шnon_trainable_variables
Щlayer_metrics
·regularization_losses
Ъmetrics
√	variables
 Ыlayer_regularization_losses
№trainable_variables
╪__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
З0"
trackable_list_wrapper
(
З0"
trackable_list_wrapper
╕
Ьlayers
Эnon_trainable_variables
Юlayer_metrics
■regularization_losses
Яmetrics
 	variables
 аlayer_regularization_losses
Аtrainable_variables
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
├
Пkernel
	Рbias
бregularization_losses
в	variables
гtrainable_variables
д	keras_api
+▌&call_and_return_all_conditional_losses
▐__call__"
_tf_keras_layer
├
Сkernel
	Тbias
еregularization_losses
ж	variables
зtrainable_variables
и	keras_api
+▀&call_and_return_all_conditional_losses
р__call__"
_tf_keras_layer
 "
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
╕
йlayers
кnon_trainable_variables
лlayer_metrics
Йregularization_losses
мmetrics
К	variables
 нlayer_regularization_losses
Лtrainable_variables
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
7
З0
И1
Y2"
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
@
в0
г1
д2
е3"
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
R

оtotal

пcount
░	variables
▒	keras_api"
_tf_keras_metric
Р
▓true_positives
│true_negatives
┤false_positives
╡false_negatives
╢	variables
╖	keras_api"
_tf_keras_metric
v
╕
thresholds
╣true_positives
║false_negatives
╗	variables
╝	keras_api"
_tf_keras_metric
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
П0
Р1"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
╕
╜layers
╛non_trainable_variables
┐layer_metrics
бregularization_losses
└metrics
в	variables
 ┴layer_regularization_losses
гtrainable_variables
▐__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
╕
┬layers
├non_trainable_variables
─layer_metrics
еregularization_losses
┼metrics
ж	variables
 ╞layer_regularization_losses
зtrainable_variables
р__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
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
:  (2total
:  (2count
0
о0
п1"
trackable_list_wrapper
.
░	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
@
▓0
│1
┤2
╡3"
trackable_list_wrapper
.
╢	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
╣0
║1"
trackable_list_wrapper
.
╗	variables"
_generic_user_object
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
N:L▌2AAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/m
M:K▌2@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/m
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/m
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/m
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/m
G:E	▌26Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/m
E:C	▐24Adam/dcn__attention__parallel__v1_1/dense_7/kernel/m
>:<22Adam/dcn__attention__parallel__v1_1/dense_7/bias/m
M:Kh2=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/m
N:L	 2=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/m
N:L	Т2=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/m
N:L	х2=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/m
N:L	┌2=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/m
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/m
N:L	╪
2=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/m
N:L	╔2=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/m
M:K
2=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/m
N:L	Г2=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/m
N:L	└2=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/m
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/m
N:L	╬2=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/m
N:L	З2=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/m
M:K
2=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/m
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/m
N:L	╕2=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/m
M:K'2=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/m
N:L	╩2=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/m
N:L	┼2=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/m
M:K*2=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/m
M:K	2=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/m
N:L	╨2=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/m
M:KC2=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/m
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/m
N:L	╕2=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/m
S:Q	▌ 2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/m
L:J 2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/m
R:P  2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/m
L:J 2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/m
R:P 2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/m
L:J2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/m
N:L▌2AAdam/dcn__attention__parallel__v1_1/batch_normalization_1/gamma/v
M:K▌2@Adam/dcn__attention__parallel__v1_1/batch_normalization_1/beta/v
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wq0/v
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wk0/v
H:F	▌27Adam/dcn__attention__parallel__v1_1/cross_layer_1/wv0/v
G:E	▌26Adam/dcn__attention__parallel__v1_1/cross_layer_1/b0/v
E:C	▐24Adam/dcn__attention__parallel__v1_1/dense_7/kernel/v
>:<22Adam/dcn__attention__parallel__v1_1/dense_7/bias/v
M:Kh2=Adam/dcn__attention__parallel__v1_1/embedding_26/embeddings/v
N:L	 2=Adam/dcn__attention__parallel__v1_1/embedding_27/embeddings/v
N:L	Т2=Adam/dcn__attention__parallel__v1_1/embedding_36/embeddings/v
N:L	х2=Adam/dcn__attention__parallel__v1_1/embedding_37/embeddings/v
N:L	┌2=Adam/dcn__attention__parallel__v1_1/embedding_38/embeddings/v
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_39/embeddings/v
N:L	╪
2=Adam/dcn__attention__parallel__v1_1/embedding_40/embeddings/v
N:L	╔2=Adam/dcn__attention__parallel__v1_1/embedding_41/embeddings/v
M:K
2=Adam/dcn__attention__parallel__v1_1/embedding_42/embeddings/v
N:L	Г2=Adam/dcn__attention__parallel__v1_1/embedding_43/embeddings/v
N:L	└2=Adam/dcn__attention__parallel__v1_1/embedding_44/embeddings/v
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_45/embeddings/v
N:L	╬2=Adam/dcn__attention__parallel__v1_1/embedding_28/embeddings/v
N:L	З2=Adam/dcn__attention__parallel__v1_1/embedding_46/embeddings/v
M:K
2=Adam/dcn__attention__parallel__v1_1/embedding_47/embeddings/v
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_48/embeddings/v
N:L	╕2=Adam/dcn__attention__parallel__v1_1/embedding_49/embeddings/v
M:K'2=Adam/dcn__attention__parallel__v1_1/embedding_50/embeddings/v
N:L	╩2=Adam/dcn__attention__parallel__v1_1/embedding_51/embeddings/v
N:L	┼2=Adam/dcn__attention__parallel__v1_1/embedding_29/embeddings/v
M:K*2=Adam/dcn__attention__parallel__v1_1/embedding_30/embeddings/v
M:K	2=Adam/dcn__attention__parallel__v1_1/embedding_31/embeddings/v
N:L	╨2=Adam/dcn__attention__parallel__v1_1/embedding_32/embeddings/v
M:KC2=Adam/dcn__attention__parallel__v1_1/embedding_33/embeddings/v
M:K2=Adam/dcn__attention__parallel__v1_1/embedding_34/embeddings/v
N:L	╕2=Adam/dcn__attention__parallel__v1_1/embedding_35/embeddings/v
S:Q	▌ 2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/kernel/v
L:J 2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_4/bias/v
R:P  2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/kernel/v
L:J 2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_5/bias/v
R:P 2BAdam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/kernel/v
L:J2@Adam/dcn__attention__parallel__v1_1/dense_layer_1/dense_6/bias/v
е2в
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64772
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_65154
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64022
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64283│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╦B╚
 __inference__wrapped_model_61990input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╣2╢
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_62878
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65243
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65332
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_63761│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▐2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65352
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65386┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_1_layer_call_fn_65399
5__inference_batch_normalization_1_layer_call_fn_65412┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
р2▌
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65436
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65460╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
-__inference_dense_layer_1_layer_call_fn_65477
-__inference_dense_layer_1_layer_call_fn_65494╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
р2▌
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65569
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65620╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
к2з
-__inference_cross_layer_1_layer_call_fn_65633
-__inference_cross_layer_1_layer_call_fn_65646╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ь2щ
B__inference_dense_7_layer_call_and_return_conditional_losses_65656в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_7_layer_call_fn_65665в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▓2п
__inference_loss_fn_0_65676П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_1_65687П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_2_65698П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓2п
__inference_loss_fn_3_65709П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╩B╟
#__inference_signature_wrapper_64404input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_26_layer_call_and_return_conditional_losses_65719в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_26_layer_call_fn_65726в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_27_layer_call_and_return_conditional_losses_65736в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_27_layer_call_fn_65743в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_28_layer_call_and_return_conditional_losses_65753в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_28_layer_call_fn_65760в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_29_layer_call_and_return_conditional_losses_65770в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_29_layer_call_fn_65777в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_30_layer_call_and_return_conditional_losses_65787в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_30_layer_call_fn_65794в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_31_layer_call_and_return_conditional_losses_65804в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_31_layer_call_fn_65811в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_32_layer_call_and_return_conditional_losses_65821в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_32_layer_call_fn_65828в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_33_layer_call_and_return_conditional_losses_65838в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_33_layer_call_fn_65845в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_34_layer_call_and_return_conditional_losses_65855в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_34_layer_call_fn_65862в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_35_layer_call_and_return_conditional_losses_65872в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_35_layer_call_fn_65879в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_36_layer_call_and_return_conditional_losses_65889в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_36_layer_call_fn_65896в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_37_layer_call_and_return_conditional_losses_65906в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_37_layer_call_fn_65913в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_38_layer_call_and_return_conditional_losses_65923в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_38_layer_call_fn_65930в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_39_layer_call_and_return_conditional_losses_65940в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_39_layer_call_fn_65947в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_40_layer_call_and_return_conditional_losses_65957в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_40_layer_call_fn_65964в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_41_layer_call_and_return_conditional_losses_65974в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_41_layer_call_fn_65981в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_42_layer_call_and_return_conditional_losses_65991в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_42_layer_call_fn_65998в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_43_layer_call_and_return_conditional_losses_66008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_43_layer_call_fn_66015в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_44_layer_call_and_return_conditional_losses_66025в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_44_layer_call_fn_66032в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_45_layer_call_and_return_conditional_losses_66042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_45_layer_call_fn_66049в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_46_layer_call_and_return_conditional_losses_66059в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_46_layer_call_fn_66066в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_47_layer_call_and_return_conditional_losses_66076в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_47_layer_call_fn_66083в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_48_layer_call_and_return_conditional_losses_66093в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_48_layer_call_fn_66100в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_49_layer_call_and_return_conditional_losses_66110в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_49_layer_call_fn_66117в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_50_layer_call_and_return_conditional_losses_66127в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_50_layer_call_fn_66134в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ё2ю
G__inference_embedding_51_layer_call_and_return_conditional_losses_66144в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_embedding_51_layer_call_fn_66151в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ═
 __inference__wrapped_model_61990и?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk0в-
&в#
!К
input_1         '
к "3к0
.
output_1"К
output_1         ╕
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65352dRSQP4в1
*в'
!К
inputs         ▌
p 
к "&в#
К
0         ▌
Ъ ╕
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65386dRSQP4в1
*в'
!К
inputs         ▌
p
к "&в#
К
0         ▌
Ъ Р
5__inference_batch_normalization_1_layer_call_fn_65399WRSQP4в1
*в'
!К
inputs         ▌
p 
к "К         ▌Р
5__inference_batch_normalization_1_layer_call_fn_65412WRSQP4в1
*в'
!К
inputs         ▌
p
к "К         ▌╝
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65569p`^bd@в=
&в#
!К
inputs         ▌
к

trainingp "&в#
К
0         ▌
Ъ ╝
H__inference_cross_layer_1_layer_call_and_return_conditional_losses_65620p`^bd@в=
&в#
!К
inputs         ▌
к

trainingp"&в#
К
0         ▌
Ъ Ф
-__inference_cross_layer_1_layer_call_fn_65633c`^bd@в=
&в#
!К
inputs         ▌
к

trainingp "К         ▌Ф
-__inference_cross_layer_1_layer_call_fn_65646c`^bd@в=
&в#
!К
inputs         ▌
к

trainingp"К         ▌№
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64022Ю?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk4в1
*в'
!К
input_1         '
p 
к "%в"
К
0         
Ъ №
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64283Ю?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk4в1
*в'
!К
input_1         '
p
к "%в"
К
0         
Ъ √
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_64772Э?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk3в0
)в&
 К
inputs         '
p 
к "%в"
К
0         
Ъ √
Y__inference_dcn__attention__parallel__v1_1_layer_call_and_return_conditional_losses_65154Э?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk3в0
)в&
 К
inputs         '
p
к "%в"
К
0         
Ъ ╘
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_62878С?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk4в1
*в'
!К
input_1         '
p 
к "К         ╘
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_63761С?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk4в1
*в'
!К
input_1         '
p
к "К         ╙
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65243Р?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk3в0
)в&
 К
inputs         '
p 
к "К         ╙
>__inference_dcn__attention__parallel__v1_1_layer_call_fn_65332Р?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk3в0
)в&
 К
inputs         '
p
к "К         г
B__inference_dense_7_layer_call_and_return_conditional_losses_65656]jk0в-
&в#
!К
inputs         ▐
к "%в"
К
0         
Ъ {
'__inference_dense_7_layer_call_fn_65665Pjk0в-
&в#
!К
inputs         ▐
к "К         ├
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65436wПРСТУФ@в=
&в#
!К
inputs         ▌
к

trainingp "%в"
К
0         
Ъ ├
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65460wПРСТУФ@в=
&в#
!К
inputs         ▌
к

trainingp"%в"
К
0         
Ъ Ы
-__inference_dense_layer_1_layer_call_fn_65477jПРСТУФ@в=
&в#
!К
inputs         ▌
к

trainingp "К         Ы
-__inference_dense_layer_1_layer_call_fn_65494jПРСТУФ@в=
&в#
!К
inputs         ▌
к

trainingp"К         в
G__inference_embedding_26_layer_call_and_return_conditional_losses_65719Wu+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_26_layer_call_fn_65726Ju+в(
!в
К
inputs         
к "К         в
G__inference_embedding_27_layer_call_and_return_conditional_losses_65736Wv+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_27_layer_call_fn_65743Jv+в(
!в
К
inputs         
к "К         г
G__inference_embedding_28_layer_call_and_return_conditional_losses_65753XБ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_28_layer_call_fn_65760KБ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_29_layer_call_and_return_conditional_losses_65770XИ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_29_layer_call_fn_65777KИ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_30_layer_call_and_return_conditional_losses_65787XЙ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_30_layer_call_fn_65794KЙ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_31_layer_call_and_return_conditional_losses_65804XК+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_31_layer_call_fn_65811KК+в(
!в
К
inputs         
к "К         г
G__inference_embedding_32_layer_call_and_return_conditional_losses_65821XЛ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_32_layer_call_fn_65828KЛ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_33_layer_call_and_return_conditional_losses_65838XМ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_33_layer_call_fn_65845KМ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_34_layer_call_and_return_conditional_losses_65855XН+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_34_layer_call_fn_65862KН+в(
!в
К
inputs         
к "К         г
G__inference_embedding_35_layer_call_and_return_conditional_losses_65872XО+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_35_layer_call_fn_65879KО+в(
!в
К
inputs         
к "К         в
G__inference_embedding_36_layer_call_and_return_conditional_losses_65889Ww+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_36_layer_call_fn_65896Jw+в(
!в
К
inputs         
к "К         в
G__inference_embedding_37_layer_call_and_return_conditional_losses_65906Wx+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_37_layer_call_fn_65913Jx+в(
!в
К
inputs         
к "К         в
G__inference_embedding_38_layer_call_and_return_conditional_losses_65923Wy+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_38_layer_call_fn_65930Jy+в(
!в
К
inputs         
к "К         в
G__inference_embedding_39_layer_call_and_return_conditional_losses_65940Wz+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_39_layer_call_fn_65947Jz+в(
!в
К
inputs         
к "К         в
G__inference_embedding_40_layer_call_and_return_conditional_losses_65957W{+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_40_layer_call_fn_65964J{+в(
!в
К
inputs         
к "К         в
G__inference_embedding_41_layer_call_and_return_conditional_losses_65974W|+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_41_layer_call_fn_65981J|+в(
!в
К
inputs         
к "К         в
G__inference_embedding_42_layer_call_and_return_conditional_losses_65991W}+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_42_layer_call_fn_65998J}+в(
!в
К
inputs         
к "К         в
G__inference_embedding_43_layer_call_and_return_conditional_losses_66008W~+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_43_layer_call_fn_66015J~+в(
!в
К
inputs         
к "К         в
G__inference_embedding_44_layer_call_and_return_conditional_losses_66025W+в(
!в
К
inputs         
к "%в"
К
0         
Ъ z
,__inference_embedding_44_layer_call_fn_66032J+в(
!в
К
inputs         
к "К         г
G__inference_embedding_45_layer_call_and_return_conditional_losses_66042XА+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_45_layer_call_fn_66049KА+в(
!в
К
inputs         
к "К         г
G__inference_embedding_46_layer_call_and_return_conditional_losses_66059XВ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_46_layer_call_fn_66066KВ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_47_layer_call_and_return_conditional_losses_66076XГ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_47_layer_call_fn_66083KГ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_48_layer_call_and_return_conditional_losses_66093XД+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_48_layer_call_fn_66100KД+в(
!в
К
inputs         
к "К         г
G__inference_embedding_49_layer_call_and_return_conditional_losses_66110XЕ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_49_layer_call_fn_66117KЕ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_50_layer_call_and_return_conditional_losses_66127XЖ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_50_layer_call_fn_66134KЖ+в(
!в
К
inputs         
к "К         г
G__inference_embedding_51_layer_call_and_return_conditional_losses_66144XЗ+в(
!в
К
inputs         
к "%в"
К
0         
Ъ {
,__inference_embedding_51_layer_call_fn_66151KЗ+в(
!в
К
inputs         
к "К         :
__inference_loss_fn_0_65676^в

в 
к "К :
__inference_loss_fn_1_65687`в

в 
к "К :
__inference_loss_fn_2_65698bв

в 
к "К :
__inference_loss_fn_3_65709dв

в 
к "К █
#__inference_signature_wrapper_64404│?uvБИЙКЛМНОwxyz{|}~АВГДЕЖЗRSQP`^bdПРСТУФjk;в8
в 
1к.
,
input_1!К
input_1         '"3к0
.
output_1"К
output_1         