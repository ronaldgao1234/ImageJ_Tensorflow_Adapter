ü6
$ė#
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
ė
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
Č
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ī
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
ķ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
I
DepthToSpace

input"T
output"T"	
Ttype"

block_sizeint(0
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
b
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
n
ResizeBicubic
images"T
size
resized_images"
Ttype:

2	"
align_cornersbool( 
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
I
SpaceToDepth

input"T
output"T"	
Ttype"

block_sizeint(0
0
Square
x"T
y"T"
Ttype:
	2	
ö
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

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
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
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.2.12
b'unknown'ģ4
¢
PlaceholderPlaceholder*
dtype0*6
shape-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
¤
Placeholder_1Placeholder*
dtype0*6
shape-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
V
Placeholder_2Placeholder*
dtype0*
shape:*
_output_shapes
:

ResizeBicubicResizeBicubicPlaceholderPlaceholder_2*
T0*
align_corners( *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ē
>Default/first_layer/filters/Initializer/truncated_normal/shapeConst*%
valueB"             *
dtype0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
:
²
=Default/first_layer/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
“
?Default/first_layer/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
¢
HDefault/first_layer/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>Default/first_layer/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Æ
<Default/first_layer/filters/Initializer/truncated_normal/mulMulHDefault/first_layer/filters/Initializer/truncated_normal/TruncatedNormal?Default/first_layer/filters/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

8Default/first_layer/filters/Initializer/truncated_normalAdd<Default/first_layer/filters/Initializer/truncated_normal/mul=Default/first_layer/filters/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ļ
Default/first_layer/filters
VariableV2*
shape: *
dtype0*
	container *
shared_name *.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

"Default/first_layer/filters/AssignAssignDefault/first_layer/filters8Default/first_layer/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ŗ
 Default/first_layer/filters/readIdentityDefault/first_layer/filters*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ø
,Default/first_layer/biases/Initializer/ConstConst*
valueB *    *
dtype0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
µ
Default/first_layer/biases
VariableV2*
shape: *
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
ņ
!Default/first_layer/biases/AssignAssignDefault/first_layer/biases,Default/first_layer/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

Default/first_layer/biases/readIdentityDefault/first_layer/biases*
T0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
ā
Conv2DConv2DPlaceholder Default/first_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 

addAddConv2DDefault/first_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
Ū
HDefault/residual_block0/conv1/filters/Initializer/truncated_normal/shapeConst*%
valueB"          "   *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*
_output_shapes
:
Ę
GDefault/residual_block0/conv1/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*
_output_shapes
: 
Č
IDefault/residual_block0/conv1/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*
_output_shapes
: 
Ą
RDefault/residual_block0/conv1/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block0/conv1/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
×
FDefault/residual_block0/conv1/filters/Initializer/truncated_normal/mulMulRDefault/residual_block0/conv1/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block0/conv1/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Å
BDefault/residual_block0/conv1/filters/Initializer/truncated_normalAddFDefault/residual_block0/conv1/filters/Initializer/truncated_normal/mulGDefault/residual_block0/conv1/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
ć
%Default/residual_block0/conv1/filters
VariableV2*
shape: "*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
µ
,Default/residual_block0/conv1/filters/AssignAssign%Default/residual_block0/conv1/filtersBDefault/residual_block0/conv1/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Č
*Default/residual_block0/conv1/filters/readIdentity%Default/residual_block0/conv1/filters*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
¼
6Default/residual_block0/conv1/biases/Initializer/ConstConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
É
$Default/residual_block0/conv1/biases
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

+Default/residual_block0/conv1/biases/AssignAssign$Default/residual_block0/conv1/biases6Default/residual_block0/conv1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
¹
)Default/residual_block0/conv1/biases/readIdentity$Default/residual_block0/conv1/biases*
T0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
ę
Conv2D_1Conv2Dadd*Default/residual_block0/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

add_1AddConv2D_1)Default/residual_block0/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
_
ReluReluadd_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
Ū
HDefault/residual_block0/conv2/filters/Initializer/truncated_normal/shapeConst*%
valueB"      "   "   *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*
_output_shapes
:
Ę
GDefault/residual_block0/conv2/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*
_output_shapes
: 
Č
IDefault/residual_block0/conv2/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*
_output_shapes
: 
Ą
RDefault/residual_block0/conv2/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block0/conv2/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
×
FDefault/residual_block0/conv2/filters/Initializer/truncated_normal/mulMulRDefault/residual_block0/conv2/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block0/conv2/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Å
BDefault/residual_block0/conv2/filters/Initializer/truncated_normalAddFDefault/residual_block0/conv2/filters/Initializer/truncated_normal/mulGDefault/residual_block0/conv2/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
ć
%Default/residual_block0/conv2/filters
VariableV2*
shape:""*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
µ
,Default/residual_block0/conv2/filters/AssignAssign%Default/residual_block0/conv2/filtersBDefault/residual_block0/conv2/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Č
*Default/residual_block0/conv2/filters/readIdentity%Default/residual_block0/conv2/filters*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
¼
6Default/residual_block0/conv2/biases/Initializer/ConstConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
É
$Default/residual_block0/conv2/biases
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

+Default/residual_block0/conv2/biases/AssignAssign$Default/residual_block0/conv2/biases6Default/residual_block0/conv2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
¹
)Default/residual_block0/conv2/biases/readIdentity$Default/residual_block0/conv2/biases*
T0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
ē
Conv2D_2Conv2DRelu*Default/residual_block0/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

add_2AddConv2D_2)Default/residual_block0/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
a
Relu_1Reluadd_2*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
}
Pad/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:
z
PadPadaddPad/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
e
add_3AddPadRelu_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
Ū
HDefault/residual_block1/conv1/filters/Initializer/truncated_normal/shapeConst*%
valueB"      "   &   *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*
_output_shapes
:
Ę
GDefault/residual_block1/conv1/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*
_output_shapes
: 
Č
IDefault/residual_block1/conv1/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*
_output_shapes
: 
Ą
RDefault/residual_block1/conv1/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block1/conv1/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
×
FDefault/residual_block1/conv1/filters/Initializer/truncated_normal/mulMulRDefault/residual_block1/conv1/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block1/conv1/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Å
BDefault/residual_block1/conv1/filters/Initializer/truncated_normalAddFDefault/residual_block1/conv1/filters/Initializer/truncated_normal/mulGDefault/residual_block1/conv1/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
ć
%Default/residual_block1/conv1/filters
VariableV2*
shape:"&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
µ
,Default/residual_block1/conv1/filters/AssignAssign%Default/residual_block1/conv1/filtersBDefault/residual_block1/conv1/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Č
*Default/residual_block1/conv1/filters/readIdentity%Default/residual_block1/conv1/filters*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
¼
6Default/residual_block1/conv1/biases/Initializer/ConstConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
É
$Default/residual_block1/conv1/biases
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

+Default/residual_block1/conv1/biases/AssignAssign$Default/residual_block1/conv1/biases6Default/residual_block1/conv1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
¹
)Default/residual_block1/conv1/biases/readIdentity$Default/residual_block1/conv1/biases*
T0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
č
Conv2D_3Conv2Dadd_3*Default/residual_block1/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

add_4AddConv2D_3)Default/residual_block1/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
a
Relu_2Reluadd_4*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
Ū
HDefault/residual_block1/conv2/filters/Initializer/truncated_normal/shapeConst*%
valueB"      &   &   *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*
_output_shapes
:
Ę
GDefault/residual_block1/conv2/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*
_output_shapes
: 
Č
IDefault/residual_block1/conv2/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*
_output_shapes
: 
Ą
RDefault/residual_block1/conv2/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block1/conv2/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
×
FDefault/residual_block1/conv2/filters/Initializer/truncated_normal/mulMulRDefault/residual_block1/conv2/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block1/conv2/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Å
BDefault/residual_block1/conv2/filters/Initializer/truncated_normalAddFDefault/residual_block1/conv2/filters/Initializer/truncated_normal/mulGDefault/residual_block1/conv2/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
ć
%Default/residual_block1/conv2/filters
VariableV2*
shape:&&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
µ
,Default/residual_block1/conv2/filters/AssignAssign%Default/residual_block1/conv2/filtersBDefault/residual_block1/conv2/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Č
*Default/residual_block1/conv2/filters/readIdentity%Default/residual_block1/conv2/filters*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
¼
6Default/residual_block1/conv2/biases/Initializer/ConstConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
É
$Default/residual_block1/conv2/biases
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

+Default/residual_block1/conv2/biases/AssignAssign$Default/residual_block1/conv2/biases6Default/residual_block1/conv2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
¹
)Default/residual_block1/conv2/biases/readIdentity$Default/residual_block1/conv2/biases*
T0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
é
Conv2D_4Conv2DRelu_2*Default/residual_block1/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

add_5AddConv2D_4)Default/residual_block1/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
a
Relu_3Reluadd_5*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

Pad_1/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_1Padadd_3Pad_1/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
g
add_6AddPad_1Relu_3*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
Ū
HDefault/residual_block2/conv1/filters/Initializer/truncated_normal/shapeConst*%
valueB"      &   ,   *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*
_output_shapes
:
Ę
GDefault/residual_block2/conv1/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*
_output_shapes
: 
Č
IDefault/residual_block2/conv1/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*
_output_shapes
: 
Ą
RDefault/residual_block2/conv1/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block2/conv1/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
×
FDefault/residual_block2/conv1/filters/Initializer/truncated_normal/mulMulRDefault/residual_block2/conv1/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block2/conv1/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Å
BDefault/residual_block2/conv1/filters/Initializer/truncated_normalAddFDefault/residual_block2/conv1/filters/Initializer/truncated_normal/mulGDefault/residual_block2/conv1/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
ć
%Default/residual_block2/conv1/filters
VariableV2*
shape:&,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
µ
,Default/residual_block2/conv1/filters/AssignAssign%Default/residual_block2/conv1/filtersBDefault/residual_block2/conv1/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Č
*Default/residual_block2/conv1/filters/readIdentity%Default/residual_block2/conv1/filters*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
¼
6Default/residual_block2/conv1/biases/Initializer/ConstConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
É
$Default/residual_block2/conv1/biases
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

+Default/residual_block2/conv1/biases/AssignAssign$Default/residual_block2/conv1/biases6Default/residual_block2/conv1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
¹
)Default/residual_block2/conv1/biases/readIdentity$Default/residual_block2/conv1/biases*
T0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
č
Conv2D_5Conv2Dadd_6*Default/residual_block2/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

add_7AddConv2D_5)Default/residual_block2/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
a
Relu_4Reluadd_7*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
Ū
HDefault/residual_block2/conv2/filters/Initializer/truncated_normal/shapeConst*%
valueB"      ,   ,   *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*
_output_shapes
:
Ę
GDefault/residual_block2/conv2/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*
_output_shapes
: 
Č
IDefault/residual_block2/conv2/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*
_output_shapes
: 
Ą
RDefault/residual_block2/conv2/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block2/conv2/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
×
FDefault/residual_block2/conv2/filters/Initializer/truncated_normal/mulMulRDefault/residual_block2/conv2/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block2/conv2/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Å
BDefault/residual_block2/conv2/filters/Initializer/truncated_normalAddFDefault/residual_block2/conv2/filters/Initializer/truncated_normal/mulGDefault/residual_block2/conv2/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
ć
%Default/residual_block2/conv2/filters
VariableV2*
shape:,,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
µ
,Default/residual_block2/conv2/filters/AssignAssign%Default/residual_block2/conv2/filtersBDefault/residual_block2/conv2/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Č
*Default/residual_block2/conv2/filters/readIdentity%Default/residual_block2/conv2/filters*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
¼
6Default/residual_block2/conv2/biases/Initializer/ConstConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
É
$Default/residual_block2/conv2/biases
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

+Default/residual_block2/conv2/biases/AssignAssign$Default/residual_block2/conv2/biases6Default/residual_block2/conv2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
¹
)Default/residual_block2/conv2/biases/readIdentity$Default/residual_block2/conv2/biases*
T0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
é
Conv2D_6Conv2DRelu_4*Default/residual_block2/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

add_8AddConv2D_6)Default/residual_block2/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
a
Relu_5Reluadd_8*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

Pad_2/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_2Padadd_6Pad_2/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
g
add_9AddPad_2Relu_5*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
Ū
HDefault/residual_block3/conv1/filters/Initializer/truncated_normal/shapeConst*%
valueB"      ,   4   *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*
_output_shapes
:
Ę
GDefault/residual_block3/conv1/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*
_output_shapes
: 
Č
IDefault/residual_block3/conv1/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*
_output_shapes
: 
Ą
RDefault/residual_block3/conv1/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block3/conv1/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
×
FDefault/residual_block3/conv1/filters/Initializer/truncated_normal/mulMulRDefault/residual_block3/conv1/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block3/conv1/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Å
BDefault/residual_block3/conv1/filters/Initializer/truncated_normalAddFDefault/residual_block3/conv1/filters/Initializer/truncated_normal/mulGDefault/residual_block3/conv1/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
ć
%Default/residual_block3/conv1/filters
VariableV2*
shape:,4*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
µ
,Default/residual_block3/conv1/filters/AssignAssign%Default/residual_block3/conv1/filtersBDefault/residual_block3/conv1/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Č
*Default/residual_block3/conv1/filters/readIdentity%Default/residual_block3/conv1/filters*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
¼
6Default/residual_block3/conv1/biases/Initializer/ConstConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
É
$Default/residual_block3/conv1/biases
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

+Default/residual_block3/conv1/biases/AssignAssign$Default/residual_block3/conv1/biases6Default/residual_block3/conv1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
¹
)Default/residual_block3/conv1/biases/readIdentity$Default/residual_block3/conv1/biases*
T0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
č
Conv2D_7Conv2Dadd_9*Default/residual_block3/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

add_10AddConv2D_7)Default/residual_block3/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
b
Relu_6Reluadd_10*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
Ū
HDefault/residual_block3/conv2/filters/Initializer/truncated_normal/shapeConst*%
valueB"      4   4   *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*
_output_shapes
:
Ę
GDefault/residual_block3/conv2/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*
_output_shapes
: 
Č
IDefault/residual_block3/conv2/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*
_output_shapes
: 
Ą
RDefault/residual_block3/conv2/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block3/conv2/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
×
FDefault/residual_block3/conv2/filters/Initializer/truncated_normal/mulMulRDefault/residual_block3/conv2/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block3/conv2/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Å
BDefault/residual_block3/conv2/filters/Initializer/truncated_normalAddFDefault/residual_block3/conv2/filters/Initializer/truncated_normal/mulGDefault/residual_block3/conv2/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
ć
%Default/residual_block3/conv2/filters
VariableV2*
shape:44*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
µ
,Default/residual_block3/conv2/filters/AssignAssign%Default/residual_block3/conv2/filtersBDefault/residual_block3/conv2/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Č
*Default/residual_block3/conv2/filters/readIdentity%Default/residual_block3/conv2/filters*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
¼
6Default/residual_block3/conv2/biases/Initializer/ConstConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
É
$Default/residual_block3/conv2/biases
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

+Default/residual_block3/conv2/biases/AssignAssign$Default/residual_block3/conv2/biases6Default/residual_block3/conv2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
¹
)Default/residual_block3/conv2/biases/readIdentity$Default/residual_block3/conv2/biases*
T0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
é
Conv2D_8Conv2DRelu_6*Default/residual_block3/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

add_11AddConv2D_8)Default/residual_block3/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
b
Relu_7Reluadd_11*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

Pad_3/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_3Padadd_9Pad_3/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
h
add_12AddPad_3Relu_7*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
Ū
HDefault/residual_block4/conv1/filters/Initializer/truncated_normal/shapeConst*%
valueB"      4   >   *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*
_output_shapes
:
Ę
GDefault/residual_block4/conv1/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*
_output_shapes
: 
Č
IDefault/residual_block4/conv1/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*
_output_shapes
: 
Ą
RDefault/residual_block4/conv1/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block4/conv1/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
×
FDefault/residual_block4/conv1/filters/Initializer/truncated_normal/mulMulRDefault/residual_block4/conv1/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block4/conv1/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Å
BDefault/residual_block4/conv1/filters/Initializer/truncated_normalAddFDefault/residual_block4/conv1/filters/Initializer/truncated_normal/mulGDefault/residual_block4/conv1/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
ć
%Default/residual_block4/conv1/filters
VariableV2*
shape:4>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
µ
,Default/residual_block4/conv1/filters/AssignAssign%Default/residual_block4/conv1/filtersBDefault/residual_block4/conv1/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Č
*Default/residual_block4/conv1/filters/readIdentity%Default/residual_block4/conv1/filters*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
¼
6Default/residual_block4/conv1/biases/Initializer/ConstConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
É
$Default/residual_block4/conv1/biases
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

+Default/residual_block4/conv1/biases/AssignAssign$Default/residual_block4/conv1/biases6Default/residual_block4/conv1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
¹
)Default/residual_block4/conv1/biases/readIdentity$Default/residual_block4/conv1/biases*
T0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
é
Conv2D_9Conv2Dadd_12*Default/residual_block4/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

add_13AddConv2D_9)Default/residual_block4/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
b
Relu_8Reluadd_13*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
Ū
HDefault/residual_block4/conv2/filters/Initializer/truncated_normal/shapeConst*%
valueB"      >   >   *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*
_output_shapes
:
Ę
GDefault/residual_block4/conv2/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*
_output_shapes
: 
Č
IDefault/residual_block4/conv2/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*
_output_shapes
: 
Ą
RDefault/residual_block4/conv2/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHDefault/residual_block4/conv2/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
×
FDefault/residual_block4/conv2/filters/Initializer/truncated_normal/mulMulRDefault/residual_block4/conv2/filters/Initializer/truncated_normal/TruncatedNormalIDefault/residual_block4/conv2/filters/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Å
BDefault/residual_block4/conv2/filters/Initializer/truncated_normalAddFDefault/residual_block4/conv2/filters/Initializer/truncated_normal/mulGDefault/residual_block4/conv2/filters/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
ć
%Default/residual_block4/conv2/filters
VariableV2*
shape:>>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
µ
,Default/residual_block4/conv2/filters/AssignAssign%Default/residual_block4/conv2/filtersBDefault/residual_block4/conv2/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Č
*Default/residual_block4/conv2/filters/readIdentity%Default/residual_block4/conv2/filters*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
¼
6Default/residual_block4/conv2/biases/Initializer/ConstConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
É
$Default/residual_block4/conv2/biases
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

+Default/residual_block4/conv2/biases/AssignAssign$Default/residual_block4/conv2/biases6Default/residual_block4/conv2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
¹
)Default/residual_block4/conv2/biases/readIdentity$Default/residual_block4/conv2/biases*
T0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
ź
	Conv2D_10Conv2DRelu_8*Default/residual_block4/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

add_14Add	Conv2D_10)Default/residual_block4/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
b
Relu_9Reluadd_14*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

Pad_4/paddingsConst*9
value0B."                             
   *
dtype0*
_output_shapes

:

Pad_4Padadd_12Pad_4/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
h
add_15AddPad_4Relu_9*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
Å
=Default/last_layer/filters/Initializer/truncated_normal/shapeConst*%
valueB"      >   K   *
dtype0*-
_class#
!loc:@Default/last_layer/filters*
_output_shapes
:
°
<Default/last_layer/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*-
_class#
!loc:@Default/last_layer/filters*
_output_shapes
: 
²
>Default/last_layer/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*-
_class#
!loc:@Default/last_layer/filters*
_output_shapes
: 

GDefault/last_layer/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormal=Default/last_layer/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
«
;Default/last_layer/filters/Initializer/truncated_normal/mulMulGDefault/last_layer/filters/Initializer/truncated_normal/TruncatedNormal>Default/last_layer/filters/Initializer/truncated_normal/stddev*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

7Default/last_layer/filters/Initializer/truncated_normalAdd;Default/last_layer/filters/Initializer/truncated_normal/mul<Default/last_layer/filters/Initializer/truncated_normal/mean*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
Ķ
Default/last_layer/filters
VariableV2*
shape:>K*
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

!Default/last_layer/filters/AssignAssignDefault/last_layer/filters7Default/last_layer/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
§
Default/last_layer/filters/readIdentityDefault/last_layer/filters*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
¦
+Default/last_layer/biases/Initializer/ConstConst*
valueBK*    *
dtype0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
³
Default/last_layer/biases
VariableV2*
shape:K*
dtype0*
	container *
shared_name *,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ī
 Default/last_layer/biases/AssignAssignDefault/last_layer/biases+Default/last_layer/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

Default/last_layer/biases/readIdentityDefault/last_layer/biases*
T0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ß
	Conv2D_11Conv2Dadd_15Default/last_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K

add_16Add	Conv2D_11Default/last_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K

DepthToSpaceDepthToSpaceadd_16*
T0*

block_size*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
×
FDefault/down_sampling_layer/filters/Initializer/truncated_normal/shapeConst*%
valueB"            *
dtype0*6
_class,
*(loc:@Default/down_sampling_layer/filters*
_output_shapes
:
Ā
EDefault/down_sampling_layer/filters/Initializer/truncated_normal/meanConst*
valueB
 *    *
dtype0*6
_class,
*(loc:@Default/down_sampling_layer/filters*
_output_shapes
: 
Ä
GDefault/down_sampling_layer/filters/Initializer/truncated_normal/stddevConst*
valueB
 *ĶĢL=*
dtype0*6
_class,
*(loc:@Default/down_sampling_layer/filters*
_output_shapes
: 
ŗ
PDefault/down_sampling_layer/filters/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFDefault/down_sampling_layer/filters/Initializer/truncated_normal/shape*

seed *
seed2 *
dtype0*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Ļ
DDefault/down_sampling_layer/filters/Initializer/truncated_normal/mulMulPDefault/down_sampling_layer/filters/Initializer/truncated_normal/TruncatedNormalGDefault/down_sampling_layer/filters/Initializer/truncated_normal/stddev*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
½
@Default/down_sampling_layer/filters/Initializer/truncated_normalAddDDefault/down_sampling_layer/filters/Initializer/truncated_normal/mulEDefault/down_sampling_layer/filters/Initializer/truncated_normal/mean*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
ß
#Default/down_sampling_layer/filters
VariableV2*
shape:*
dtype0*
	container *
shared_name *6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
­
*Default/down_sampling_layer/filters/AssignAssign#Default/down_sampling_layer/filters@Default/down_sampling_layer/filters/Initializer/truncated_normal*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Ā
(Default/down_sampling_layer/filters/readIdentity#Default/down_sampling_layer/filters*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
ø
4Default/down_sampling_layer/biases/Initializer/ConstConst*
valueB*    *
dtype0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
Å
"Default/down_sampling_layer/biases
VariableV2*
shape:*
dtype0*
	container *
shared_name *5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

)Default/down_sampling_layer/biases/AssignAssign"Default/down_sampling_layer/biases4Default/down_sampling_layer/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
³
'Default/down_sampling_layer/biases/readIdentity"Default/down_sampling_layer/biases*
T0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
ī
	Conv2D_12Conv2DDepthToSpace(Default/down_sampling_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

add_17Add	Conv2D_12'Default/down_sampling_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
¤
Placeholder_3Placeholder*
dtype0*6
shape-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
¤
Placeholder_4Placeholder*
dtype0*6
shape-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
V
Placeholder_5Placeholder*
dtype0*
shape:*
_output_shapes
:

ResizeBicubic_1ResizeBicubicPlaceholder_3Placeholder_5*
T0*
align_corners( *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ē
	Conv2D_13Conv2DPlaceholder_3 Default/first_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 

add_18Add	Conv2D_13Default/first_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
ź
	Conv2D_14Conv2Dadd_18*Default/residual_block0/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

add_19Add	Conv2D_14)Default/residual_block0/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
c
Relu_10Reluadd_19*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
ė
	Conv2D_15Conv2DRelu_10*Default/residual_block0/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

add_20Add	Conv2D_15)Default/residual_block0/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
c
Relu_11Reluadd_20*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

Pad_5/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_5Padadd_18Pad_5/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
i
add_21AddPad_5Relu_11*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
ź
	Conv2D_16Conv2Dadd_21*Default/residual_block1/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

add_22Add	Conv2D_16)Default/residual_block1/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
c
Relu_12Reluadd_22*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
ė
	Conv2D_17Conv2DRelu_12*Default/residual_block1/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

add_23Add	Conv2D_17)Default/residual_block1/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
c
Relu_13Reluadd_23*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

Pad_6/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_6Padadd_21Pad_6/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
i
add_24AddPad_6Relu_13*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
ź
	Conv2D_18Conv2Dadd_24*Default/residual_block2/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

add_25Add	Conv2D_18)Default/residual_block2/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
c
Relu_14Reluadd_25*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
ė
	Conv2D_19Conv2DRelu_14*Default/residual_block2/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

add_26Add	Conv2D_19)Default/residual_block2/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
c
Relu_15Reluadd_26*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

Pad_7/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_7Padadd_24Pad_7/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
i
add_27AddPad_7Relu_15*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
ź
	Conv2D_20Conv2Dadd_27*Default/residual_block3/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

add_28Add	Conv2D_20)Default/residual_block3/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
c
Relu_16Reluadd_28*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
ė
	Conv2D_21Conv2DRelu_16*Default/residual_block3/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

add_29Add	Conv2D_21)Default/residual_block3/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
c
Relu_17Reluadd_29*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

Pad_8/paddingsConst*9
value0B."                                *
dtype0*
_output_shapes

:

Pad_8Padadd_27Pad_8/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
i
add_30AddPad_8Relu_17*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
ź
	Conv2D_22Conv2Dadd_30*Default/residual_block4/conv1/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

add_31Add	Conv2D_22)Default/residual_block4/conv1/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
c
Relu_18Reluadd_31*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
ė
	Conv2D_23Conv2DRelu_18*Default/residual_block4/conv2/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

add_32Add	Conv2D_23)Default/residual_block4/conv2/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
c
Relu_19Reluadd_32*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

Pad_9/paddingsConst*9
value0B."                             
   *
dtype0*
_output_shapes

:

Pad_9Padadd_30Pad_9/paddings*
T0*
	Tpaddings0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
i
add_33AddPad_9Relu_19*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
ß
	Conv2D_24Conv2Dadd_33Default/last_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K

add_34Add	Conv2D_24Default/last_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K

DepthToSpace_1DepthToSpaceadd_34*
T0*

block_size*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
š
	Conv2D_25Conv2DDepthToSpace_1(Default/down_sampling_layer/filters/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

add_35Add	Conv2D_25'Default/down_sampling_layer/biases/read*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
l
strided_slice/stackConst*%
valueB"                *
dtype0*
_output_shapes
:
n
strided_slice/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
n
strided_slice/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
„
strided_sliceStridedSliceadd_35strided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

ConstConst*E
value<B:"$  æ      ?   Ą       @  æ      ?*
dtype0*&
_output_shapes
:
Ģ
	Conv2D_26Conv2Dstrided_sliceConst*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
strided_slice_1/stackConst*%
valueB"                *
dtype0*
_output_shapes
:
p
strided_slice_1/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_1/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
­
strided_slice_1StridedSliceadd_35strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

Const_1Const*E
value<B:"$  æ   Ą  æ              ?   @  ?*
dtype0*&
_output_shapes
:
Š
	Conv2D_27Conv2Dstrided_slice_1Const_1*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
l
mulMul	Conv2D_26	Conv2D_26*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
mul_1Mul	Conv2D_27	Conv2D_27*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
e
add_36Addmulmul_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
Const_2Const*%
valueB"             *
dtype0*
_output_shapes
:
[
MeanMeanadd_36Const_2*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
n
strided_slice_2/stackConst*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_2/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_2/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
­
strided_slice_2StridedSliceadd_35strided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

Const_3Const*E
value<B:"$  æ      ?   Ą       @  æ      ?*
dtype0*&
_output_shapes
:
Š
	Conv2D_28Conv2Dstrided_slice_2Const_3*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
strided_slice_3/stackConst*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_3/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_3/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
­
strided_slice_3StridedSliceadd_35strided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

Const_4Const*E
value<B:"$  æ   Ą  æ              ?   @  ?*
dtype0*&
_output_shapes
:
Š
	Conv2D_29Conv2Dstrided_slice_3Const_4*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
mul_2Mul	Conv2D_28	Conv2D_28*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
mul_3Mul	Conv2D_29	Conv2D_29*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
g
add_37Addmul_2mul_3*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
Const_5Const*%
valueB"             *
dtype0*
_output_shapes
:
]
Mean_1Meanadd_37Const_5*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
<
add_38AddMeanMean_1*
T0*
_output_shapes
: 
n
strided_slice_4/stackConst*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_4/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_4/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
­
strided_slice_4StridedSliceadd_35strided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

Const_6Const*E
value<B:"$  æ      ?   Ą       @  æ      ?*
dtype0*&
_output_shapes
:
Š
	Conv2D_30Conv2Dstrided_slice_4Const_6*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
strided_slice_5/stackConst*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_5/stack_1Const*%
valueB"               *
dtype0*
_output_shapes
:
p
strided_slice_5/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
­
strided_slice_5StridedSliceadd_35strided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

Const_7Const*E
value<B:"$  æ   Ą  æ              ?   @  ?*
dtype0*&
_output_shapes
:
Š
	Conv2D_31Conv2Dstrided_slice_5Const_7*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
mul_4Mul	Conv2D_30	Conv2D_30*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
n
mul_5Mul	Conv2D_31	Conv2D_31*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
g
add_39Addmul_4mul_5*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
Const_8Const*%
valueB"             *
dtype0*
_output_shapes
:
]
Mean_2Meanadd_39Const_8*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
>
add_40Addadd_38Mean_2*
T0*
_output_shapes
: 
N
	truediv/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
F
truedivRealDivadd_40	truediv/y*
T0*
_output_shapes
: 
m
subSubPlaceholder_4add_35*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
a
SquareSquaresub*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
Const_9Const*%
valueB"             *
dtype0*
_output_shapes
:
]
Mean_3MeanSquareConst_9*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
L
mul_6/xConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
mul_6Mulmul_6/xtruediv*
T0*
_output_shapes
: 
=
add_41AddMean_3mul_6*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
T
gradients/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
^
gradients/add_41_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
gradients/add_41_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
½
+gradients/add_41_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_41_grad/Shapegradients/add_41_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_41_grad/SumSumgradients/Fill+gradients/add_41_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_41_grad/ReshapeReshapegradients/add_41_grad/Sumgradients/add_41_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
”
gradients/add_41_grad/Sum_1Sumgradients/Fill-gradients/add_41_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_41_grad/Reshape_1Reshapegradients/add_41_grad/Sum_1gradients/add_41_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_41_grad/tuple/group_depsNoOp^gradients/add_41_grad/Reshape ^gradients/add_41_grad/Reshape_1
Õ
.gradients/add_41_grad/tuple/control_dependencyIdentitygradients/add_41_grad/Reshape'^gradients/add_41_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_41_grad/Reshape*
_output_shapes
: 
Ū
0gradients/add_41_grad/tuple/control_dependency_1Identitygradients/add_41_grad/Reshape_1'^gradients/add_41_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_41_grad/Reshape_1*
_output_shapes
: 
|
#gradients/Mean_3_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
¼
gradients/Mean_3_grad/ReshapeReshape.gradients/add_41_grad/tuple/control_dependency#gradients/Mean_3_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
a
gradients/Mean_3_grad/ShapeShapeSquare*
T0*
out_type0*
_output_shapes
:
¼
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*
T0*

Tmultiples0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/Mean_3_grad/Shape_1ShapeSquare*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_3_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
g
gradients/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
a
gradients/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
r
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
¬
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
]
gradients/mul_6_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
_
gradients/mul_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ŗ
*gradients/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_6_grad/Shapegradients/mul_6_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
{
gradients/mul_6_grad/mulMul0gradients/add_41_grad/tuple/control_dependency_1truediv*
T0*
_output_shapes
: 
„
gradients/mul_6_grad/SumSumgradients/mul_6_grad/mul*gradients/mul_6_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/mul_6_grad/ReshapeReshapegradients/mul_6_grad/Sumgradients/mul_6_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
}
gradients/mul_6_grad/mul_1Mulmul_6/x0gradients/add_41_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
«
gradients/mul_6_grad/Sum_1Sumgradients/mul_6_grad/mul_1,gradients/mul_6_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/mul_6_grad/Reshape_1Reshapegradients/mul_6_grad/Sum_1gradients/mul_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients/mul_6_grad/tuple/group_depsNoOp^gradients/mul_6_grad/Reshape^gradients/mul_6_grad/Reshape_1
Ń
-gradients/mul_6_grad/tuple/control_dependencyIdentitygradients/mul_6_grad/Reshape&^gradients/mul_6_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_6_grad/Reshape*
_output_shapes
: 
×
/gradients/mul_6_grad/tuple/control_dependency_1Identitygradients/mul_6_grad/Reshape_1&^gradients/mul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_6_grad/Reshape_1*
_output_shapes
: 

gradients/Square_grad/mul/xConst^gradients/Mean_3_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 

gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ø
gradients/Square_grad/mul_1Mulgradients/Mean_3_grad/truedivgradients/Square_grad/mul*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
_
gradients/truediv_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
a
gradients/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ą
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/truediv_grad/RealDivRealDiv/gradients/mul_6_grad/tuple/control_dependency_1	truediv/y*
T0*
_output_shapes
: 
Æ
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
J
gradients/truediv_grad/NegNegadd_40*
T0*
_output_shapes
: 
s
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/Neg	truediv/y*
T0*
_output_shapes
: 
y
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1	truediv/y*
T0*
_output_shapes
: 

gradients/truediv_grad/mulMul/gradients/mul_6_grad/tuple/control_dependency_1 gradients/truediv_grad/RealDiv_2*
T0*
_output_shapes
: 
Æ
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
Ł
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
_output_shapes
: 
ß
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1*
_output_shapes
: 
e
gradients/sub_grad/ShapeShapePlaceholder_4*
T0*
out_type0*
_output_shapes
:
`
gradients/sub_grad/Shape_1Shapeadd_35*
T0*
out_type0*
_output_shapes
:
“
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¤
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
±
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ø
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
µ
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
ō
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ś
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
^
gradients/add_40_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
gradients/add_40_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
½
+gradients/add_40_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_40_grad/Shapegradients/add_40_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¾
gradients/add_40_grad/SumSum/gradients/truediv_grad/tuple/control_dependency+gradients/add_40_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_40_grad/ReshapeReshapegradients/add_40_grad/Sumgradients/add_40_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Ā
gradients/add_40_grad/Sum_1Sum/gradients/truediv_grad/tuple/control_dependency-gradients/add_40_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_40_grad/Reshape_1Reshapegradients/add_40_grad/Sum_1gradients/add_40_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_40_grad/tuple/group_depsNoOp^gradients/add_40_grad/Reshape ^gradients/add_40_grad/Reshape_1
Õ
.gradients/add_40_grad/tuple/control_dependencyIdentitygradients/add_40_grad/Reshape'^gradients/add_40_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_40_grad/Reshape*
_output_shapes
: 
Ū
0gradients/add_40_grad/tuple/control_dependency_1Identitygradients/add_40_grad/Reshape_1'^gradients/add_40_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_40_grad/Reshape_1*
_output_shapes
: 
^
gradients/add_38_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
`
gradients/add_38_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
½
+gradients/add_38_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_38_grad/Shapegradients/add_38_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
½
gradients/add_38_grad/SumSum.gradients/add_40_grad/tuple/control_dependency+gradients/add_38_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_38_grad/ReshapeReshapegradients/add_38_grad/Sumgradients/add_38_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Į
gradients/add_38_grad/Sum_1Sum.gradients/add_40_grad/tuple/control_dependency-gradients/add_38_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_38_grad/Reshape_1Reshapegradients/add_38_grad/Sum_1gradients/add_38_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_38_grad/tuple/group_depsNoOp^gradients/add_38_grad/Reshape ^gradients/add_38_grad/Reshape_1
Õ
.gradients/add_38_grad/tuple/control_dependencyIdentitygradients/add_38_grad/Reshape'^gradients/add_38_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_38_grad/Reshape*
_output_shapes
: 
Ū
0gradients/add_38_grad/tuple/control_dependency_1Identitygradients/add_38_grad/Reshape_1'^gradients/add_38_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_38_grad/Reshape_1*
_output_shapes
: 
|
#gradients/Mean_2_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
¾
gradients/Mean_2_grad/ReshapeReshape0gradients/add_40_grad/tuple/control_dependency_1#gradients/Mean_2_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
a
gradients/Mean_2_grad/ShapeShapeadd_39*
T0*
out_type0*
_output_shapes
:
¼
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*
T0*

Tmultiples0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/Mean_2_grad/Shape_1Shapeadd_39*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
g
gradients/Mean_2_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
a
gradients/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
r
gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
¬
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
z
!gradients/Mean_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
ø
gradients/Mean_grad/ReshapeReshape.gradients/add_38_grad/tuple/control_dependency!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
_
gradients/Mean_grad/ShapeShapeadd_36*
T0*
out_type0*
_output_shapes
:
¶
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
a
gradients/Mean_grad/Shape_1Shapeadd_36*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
¦
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
|
#gradients/Mean_1_grad/Reshape/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
¾
gradients/Mean_1_grad/ReshapeReshape0gradients/add_38_grad/tuple/control_dependency_1#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0*&
_output_shapes
:
a
gradients/Mean_1_grad/ShapeShapeadd_37*
T0*
out_type0*
_output_shapes
:
¼
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*
T0*

Tmultiples0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/Mean_1_grad/Shape_1Shapeadd_37*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
g
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
 
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
r
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*

DstT0*
_output_shapes
: 
¬
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
gradients/add_39_grad/ShapeShapemul_4*
T0*
out_type0*
_output_shapes
:
b
gradients/add_39_grad/Shape_1Shapemul_5*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_39_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_39_grad/Shapegradients/add_39_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¬
gradients/add_39_grad/SumSumgradients/Mean_2_grad/truediv+gradients/add_39_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_39_grad/ReshapeReshapegradients/add_39_grad/Sumgradients/add_39_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
°
gradients/add_39_grad/Sum_1Sumgradients/Mean_2_grad/truediv-gradients/add_39_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_39_grad/Reshape_1Reshapegradients/add_39_grad/Sum_1gradients/add_39_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
&gradients/add_39_grad/tuple/group_depsNoOp^gradients/add_39_grad/Reshape ^gradients/add_39_grad/Reshape_1

.gradients/add_39_grad/tuple/control_dependencyIdentitygradients/add_39_grad/Reshape'^gradients/add_39_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_39_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

0gradients/add_39_grad/tuple/control_dependency_1Identitygradients/add_39_grad/Reshape_1'^gradients/add_39_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_39_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
^
gradients/add_36_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
b
gradients/add_36_grad/Shape_1Shapemul_1*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_36_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_36_grad/Shapegradients/add_36_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ŗ
gradients/add_36_grad/SumSumgradients/Mean_grad/truediv+gradients/add_36_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_36_grad/ReshapeReshapegradients/add_36_grad/Sumgradients/add_36_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
®
gradients/add_36_grad/Sum_1Sumgradients/Mean_grad/truediv-gradients/add_36_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_36_grad/Reshape_1Reshapegradients/add_36_grad/Sum_1gradients/add_36_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
&gradients/add_36_grad/tuple/group_depsNoOp^gradients/add_36_grad/Reshape ^gradients/add_36_grad/Reshape_1

.gradients/add_36_grad/tuple/control_dependencyIdentitygradients/add_36_grad/Reshape'^gradients/add_36_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_36_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

0gradients/add_36_grad/tuple/control_dependency_1Identitygradients/add_36_grad/Reshape_1'^gradients/add_36_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_36_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
`
gradients/add_37_grad/ShapeShapemul_2*
T0*
out_type0*
_output_shapes
:
b
gradients/add_37_grad/Shape_1Shapemul_3*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_37_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_37_grad/Shapegradients/add_37_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¬
gradients/add_37_grad/SumSumgradients/Mean_1_grad/truediv+gradients/add_37_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_37_grad/ReshapeReshapegradients/add_37_grad/Sumgradients/add_37_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
°
gradients/add_37_grad/Sum_1Sumgradients/Mean_1_grad/truediv-gradients/add_37_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_37_grad/Reshape_1Reshapegradients/add_37_grad/Sum_1gradients/add_37_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
p
&gradients/add_37_grad/tuple/group_depsNoOp^gradients/add_37_grad/Reshape ^gradients/add_37_grad/Reshape_1

.gradients/add_37_grad/tuple/control_dependencyIdentitygradients/add_37_grad/Reshape'^gradients/add_37_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_37_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

0gradients/add_37_grad/tuple/control_dependency_1Identitygradients/add_37_grad/Reshape_1'^gradients/add_37_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_37_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/mul_4_grad/ShapeShape	Conv2D_30*
T0*
out_type0*
_output_shapes
:
e
gradients/mul_4_grad/Shape_1Shape	Conv2D_30*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_4_grad/Shapegradients/mul_4_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¦
gradients/mul_4_grad/mulMul.gradients/add_39_grad/tuple/control_dependency	Conv2D_30*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_4_grad/SumSumgradients/mul_4_grad/mul*gradients/mul_4_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_4_grad/ReshapeReshapegradients/mul_4_grad/Sumgradients/mul_4_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ø
gradients/mul_4_grad/mul_1Mul	Conv2D_30.gradients/add_39_grad/tuple/control_dependency*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
«
gradients/mul_4_grad/Sum_1Sumgradients/mul_4_grad/mul_1,gradients/mul_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
½
gradients/mul_4_grad/Reshape_1Reshapegradients/mul_4_grad/Sum_1gradients/mul_4_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Reshape^gradients/mul_4_grad/Reshape_1
ü
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Reshape&^gradients/mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_4_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Reshape_1&^gradients/mul_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_4_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/mul_5_grad/ShapeShape	Conv2D_31*
T0*
out_type0*
_output_shapes
:
e
gradients/mul_5_grad/Shape_1Shape	Conv2D_31*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_5_grad/Shapegradients/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ø
gradients/mul_5_grad/mulMul0gradients/add_39_grad/tuple/control_dependency_1	Conv2D_31*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_5_grad/SumSumgradients/mul_5_grad/mul*gradients/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_5_grad/ReshapeReshapegradients/mul_5_grad/Sumgradients/mul_5_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ
gradients/mul_5_grad/mul_1Mul	Conv2D_310gradients/add_39_grad/tuple/control_dependency_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
«
gradients/mul_5_grad/Sum_1Sumgradients/mul_5_grad/mul_1,gradients/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
½
gradients/mul_5_grad/Reshape_1Reshapegradients/mul_5_grad/Sum_1gradients/mul_5_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Reshape^gradients/mul_5_grad/Reshape_1
ü
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Reshape&^gradients/mul_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_5_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Reshape_1&^gradients/mul_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_5_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
a
gradients/mul_grad/ShapeShape	Conv2D_26*
T0*
out_type0*
_output_shapes
:
c
gradients/mul_grad/Shape_1Shape	Conv2D_26*
T0*
out_type0*
_output_shapes
:
“
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¤
gradients/mul_grad/mulMul.gradients/add_36_grad/tuple/control_dependency	Conv2D_26*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
±
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
¦
gradients/mul_grad/mul_1Mul	Conv2D_26.gradients/add_36_grad/tuple/control_dependency*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ō
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ś
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/mul_1_grad/ShapeShape	Conv2D_27*
T0*
out_type0*
_output_shapes
:
e
gradients/mul_1_grad/Shape_1Shape	Conv2D_27*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ø
gradients/mul_1_grad/mulMul0gradients/add_36_grad/tuple/control_dependency_1	Conv2D_27*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ
gradients/mul_1_grad/mul_1Mul	Conv2D_270gradients/add_36_grad/tuple/control_dependency_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
«
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
½
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
ü
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/mul_2_grad/ShapeShape	Conv2D_28*
T0*
out_type0*
_output_shapes
:
e
gradients/mul_2_grad/Shape_1Shape	Conv2D_28*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¦
gradients/mul_2_grad/mulMul.gradients/add_37_grad/tuple/control_dependency	Conv2D_28*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ø
gradients/mul_2_grad/mul_1Mul	Conv2D_28.gradients/add_37_grad/tuple/control_dependency*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
«
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
½
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
ü
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
c
gradients/mul_3_grad/ShapeShape	Conv2D_29*
T0*
out_type0*
_output_shapes
:
e
gradients/mul_3_grad/Shape_1Shape	Conv2D_29*
T0*
out_type0*
_output_shapes
:
ŗ
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ø
gradients/mul_3_grad/mulMul0gradients/add_37_grad/tuple/control_dependency_1	Conv2D_29*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
„
gradients/mul_3_grad/SumSumgradients/mul_3_grad/mul*gradients/mul_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
·
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ
gradients/mul_3_grad/mul_1Mul	Conv2D_290gradients/add_37_grad/tuple/control_dependency_1*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
«
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
½
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
ü
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ü
gradients/AddNAddN-gradients/mul_4_grad/tuple/control_dependency/gradients/mul_4_grad/tuple/control_dependency_1*
N*
T0*/
_class%
#!loc:@gradients/mul_4_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
gradients/Conv2D_30_grad/ShapeShapestrided_slice_4*
T0*
out_type0*
_output_shapes
:
Ø
,gradients/Conv2D_30_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_30_grad/ShapeConst_6gradients/AddN*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_30_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_30_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice_4 gradients/Conv2D_30_grad/Shape_1gradients/AddN*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_30_grad/tuple/group_depsNoOp-^gradients/Conv2D_30_grad/Conv2DBackpropInput.^gradients/Conv2D_30_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_30_grad/tuple/control_dependencyIdentity,gradients/Conv2D_30_grad/Conv2DBackpropInput*^gradients/Conv2D_30_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_30_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_30_grad/tuple/control_dependency_1Identity-gradients/Conv2D_30_grad/Conv2DBackpropFilter*^gradients/Conv2D_30_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_30_grad/Conv2DBackpropFilter*&
_output_shapes
:
ž
gradients/AddN_1AddN-gradients/mul_5_grad/tuple/control_dependency/gradients/mul_5_grad/tuple/control_dependency_1*
N*
T0*/
_class%
#!loc:@gradients/mul_5_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
gradients/Conv2D_31_grad/ShapeShapestrided_slice_5*
T0*
out_type0*
_output_shapes
:
Ŗ
,gradients/Conv2D_31_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_31_grad/ShapeConst_7gradients/AddN_1*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_31_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_31_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice_5 gradients/Conv2D_31_grad/Shape_1gradients/AddN_1*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_31_grad/tuple/group_depsNoOp-^gradients/Conv2D_31_grad/Conv2DBackpropInput.^gradients/Conv2D_31_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_31_grad/tuple/control_dependencyIdentity,gradients/Conv2D_31_grad/Conv2DBackpropInput*^gradients/Conv2D_31_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_31_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_31_grad/tuple/control_dependency_1Identity-gradients/Conv2D_31_grad/Conv2DBackpropFilter*^gradients/Conv2D_31_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_31_grad/Conv2DBackpropFilter*&
_output_shapes
:
ų
gradients/AddN_2AddN+gradients/mul_grad/tuple/control_dependency-gradients/mul_grad/tuple/control_dependency_1*
N*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
k
gradients/Conv2D_26_grad/ShapeShapestrided_slice*
T0*
out_type0*
_output_shapes
:
Ø
,gradients/Conv2D_26_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_26_grad/ShapeConstgradients/AddN_2*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_26_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_26_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice gradients/Conv2D_26_grad/Shape_1gradients/AddN_2*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_26_grad/tuple/group_depsNoOp-^gradients/Conv2D_26_grad/Conv2DBackpropInput.^gradients/Conv2D_26_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_26_grad/tuple/control_dependencyIdentity,gradients/Conv2D_26_grad/Conv2DBackpropInput*^gradients/Conv2D_26_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_26_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_26_grad/tuple/control_dependency_1Identity-gradients/Conv2D_26_grad/Conv2DBackpropFilter*^gradients/Conv2D_26_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_26_grad/Conv2DBackpropFilter*&
_output_shapes
:
ž
gradients/AddN_3AddN-gradients/mul_1_grad/tuple/control_dependency/gradients/mul_1_grad/tuple/control_dependency_1*
N*
T0*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
gradients/Conv2D_27_grad/ShapeShapestrided_slice_1*
T0*
out_type0*
_output_shapes
:
Ŗ
,gradients/Conv2D_27_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_27_grad/ShapeConst_1gradients/AddN_3*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_27_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_27_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice_1 gradients/Conv2D_27_grad/Shape_1gradients/AddN_3*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_27_grad/tuple/group_depsNoOp-^gradients/Conv2D_27_grad/Conv2DBackpropInput.^gradients/Conv2D_27_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_27_grad/tuple/control_dependencyIdentity,gradients/Conv2D_27_grad/Conv2DBackpropInput*^gradients/Conv2D_27_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_27_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_27_grad/tuple/control_dependency_1Identity-gradients/Conv2D_27_grad/Conv2DBackpropFilter*^gradients/Conv2D_27_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_27_grad/Conv2DBackpropFilter*&
_output_shapes
:
ž
gradients/AddN_4AddN-gradients/mul_2_grad/tuple/control_dependency/gradients/mul_2_grad/tuple/control_dependency_1*
N*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
gradients/Conv2D_28_grad/ShapeShapestrided_slice_2*
T0*
out_type0*
_output_shapes
:
Ŗ
,gradients/Conv2D_28_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_28_grad/ShapeConst_3gradients/AddN_4*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_28_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_28_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice_2 gradients/Conv2D_28_grad/Shape_1gradients/AddN_4*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_28_grad/tuple/group_depsNoOp-^gradients/Conv2D_28_grad/Conv2DBackpropInput.^gradients/Conv2D_28_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_28_grad/tuple/control_dependencyIdentity,gradients/Conv2D_28_grad/Conv2DBackpropInput*^gradients/Conv2D_28_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_28_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_28_grad/tuple/control_dependency_1Identity-gradients/Conv2D_28_grad/Conv2DBackpropFilter*^gradients/Conv2D_28_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_28_grad/Conv2DBackpropFilter*&
_output_shapes
:
ž
gradients/AddN_5AddN-gradients/mul_3_grad/tuple/control_dependency/gradients/mul_3_grad/tuple/control_dependency_1*
N*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
m
gradients/Conv2D_29_grad/ShapeShapestrided_slice_3*
T0*
out_type0*
_output_shapes
:
Ŗ
,gradients/Conv2D_29_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_29_grad/ShapeConst_4gradients/AddN_5*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_29_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:

-gradients/Conv2D_29_grad/Conv2DBackpropFilterConv2DBackpropFilterstrided_slice_3 gradients/Conv2D_29_grad/Shape_1gradients/AddN_5*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_29_grad/tuple/group_depsNoOp-^gradients/Conv2D_29_grad/Conv2DBackpropInput.^gradients/Conv2D_29_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_29_grad/tuple/control_dependencyIdentity,gradients/Conv2D_29_grad/Conv2DBackpropInput*^gradients/Conv2D_29_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_29_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_29_grad/tuple/control_dependency_1Identity-gradients/Conv2D_29_grad/Conv2DBackpropFilter*^gradients/Conv2D_29_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_29_grad/Conv2DBackpropFilter*&
_output_shapes
:
j
$gradients/strided_slice_4_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:
¢
/gradients/strided_slice_4_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_4_grad/Shapestrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_21gradients/Conv2D_30_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
j
$gradients/strided_slice_5_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:
¢
/gradients/strided_slice_5_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_5_grad/Shapestrided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_21gradients/Conv2D_31_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
h
"gradients/strided_slice_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:

-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad"gradients/strided_slice_grad/Shapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_21gradients/Conv2D_26_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
j
$gradients/strided_slice_1_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:
¢
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_1_grad/Shapestrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_21gradients/Conv2D_27_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
j
$gradients/strided_slice_2_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:
¢
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_2_grad/Shapestrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_21gradients/Conv2D_28_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
j
$gradients/strided_slice_3_grad/ShapeShapeadd_35*
T0*
out_type0*
_output_shapes
:
¢
/gradients/strided_slice_3_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_3_grad/Shapestrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_21gradients/Conv2D_29_grad/tuple/control_dependency*
T0*
Index0*

begin_mask*
end_mask*
ellipsis_mask *
new_axis_mask *
shrink_axis_mask *A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ń
gradients/AddN_6AddN-gradients/sub_grad/tuple/control_dependency_1/gradients/strided_slice_4_grad/StridedSliceGrad/gradients/strided_slice_5_grad/StridedSliceGrad-gradients/strided_slice_grad/StridedSliceGrad/gradients/strided_slice_1_grad/StridedSliceGrad/gradients/strided_slice_2_grad/StridedSliceGrad/gradients/strided_slice_3_grad/StridedSliceGrad*
N*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
d
gradients/add_35_grad/ShapeShape	Conv2D_25*
T0*
out_type0*
_output_shapes
:
g
gradients/add_35_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
½
+gradients/add_35_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_35_grad/Shapegradients/add_35_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_35_grad/SumSumgradients/AddN_6+gradients/add_35_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_35_grad/ReshapeReshapegradients/add_35_grad/Sumgradients/add_35_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
£
gradients/add_35_grad/Sum_1Sumgradients/AddN_6-gradients/add_35_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_35_grad/Reshape_1Reshapegradients/add_35_grad/Sum_1gradients/add_35_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
p
&gradients/add_35_grad/tuple/group_depsNoOp^gradients/add_35_grad/Reshape ^gradients/add_35_grad/Reshape_1

.gradients/add_35_grad/tuple/control_dependencyIdentitygradients/add_35_grad/Reshape'^gradients/add_35_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_35_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’
ß
0gradients/add_35_grad/tuple/control_dependency_1Identitygradients/add_35_grad/Reshape_1'^gradients/add_35_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_35_grad/Reshape_1*
_output_shapes
:
l
gradients/Conv2D_25_grad/ShapeShapeDepthToSpace_1*
T0*
out_type0*
_output_shapes
:
é
,gradients/Conv2D_25_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_25_grad/Shape(Default/down_sampling_layer/filters/read.gradients/add_35_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_25_grad/Shape_1Const*%
valueB"            *
dtype0*
_output_shapes
:
Æ
-gradients/Conv2D_25_grad/Conv2DBackpropFilterConv2DBackpropFilterDepthToSpace_1 gradients/Conv2D_25_grad/Shape_1.gradients/add_35_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:

)gradients/Conv2D_25_grad/tuple/group_depsNoOp-^gradients/Conv2D_25_grad/Conv2DBackpropInput.^gradients/Conv2D_25_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_25_grad/tuple/control_dependencyIdentity,gradients/Conv2D_25_grad/Conv2DBackpropInput*^gradients/Conv2D_25_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_25_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_25_grad/tuple/control_dependency_1Identity-gradients/Conv2D_25_grad/Conv2DBackpropFilter*^gradients/Conv2D_25_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_25_grad/Conv2DBackpropFilter*&
_output_shapes
:
Ė
*gradients/DepthToSpace_1_grad/SpaceToDepthSpaceToDepth1gradients/Conv2D_25_grad/tuple/control_dependency*
T0*

block_size*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K
d
gradients/add_34_grad/ShapeShape	Conv2D_24*
T0*
out_type0*
_output_shapes
:
g
gradients/add_34_grad/Shape_1Const*
valueB:K*
dtype0*
_output_shapes
:
½
+gradients/add_34_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_34_grad/Shapegradients/add_34_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
¹
gradients/add_34_grad/SumSum*gradients/DepthToSpace_1_grad/SpaceToDepth+gradients/add_34_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_34_grad/ReshapeReshapegradients/add_34_grad/Sumgradients/add_34_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K
½
gradients/add_34_grad/Sum_1Sum*gradients/DepthToSpace_1_grad/SpaceToDepth-gradients/add_34_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_34_grad/Reshape_1Reshapegradients/add_34_grad/Sum_1gradients/add_34_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:K
p
&gradients/add_34_grad/tuple/group_depsNoOp^gradients/add_34_grad/Reshape ^gradients/add_34_grad/Reshape_1

.gradients/add_34_grad/tuple/control_dependencyIdentitygradients/add_34_grad/Reshape'^gradients/add_34_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_34_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’K
ß
0gradients/add_34_grad/tuple/control_dependency_1Identitygradients/add_34_grad/Reshape_1'^gradients/add_34_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_34_grad/Reshape_1*
_output_shapes
:K
d
gradients/Conv2D_24_grad/ShapeShapeadd_33*
T0*
out_type0*
_output_shapes
:
ą
,gradients/Conv2D_24_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_24_grad/ShapeDefault/last_layer/filters/read.gradients/add_34_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_24_grad/Shape_1Const*%
valueB"      >   K   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_24_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_33 gradients/Conv2D_24_grad/Shape_1.gradients/add_34_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:>K

)gradients/Conv2D_24_grad/tuple/group_depsNoOp-^gradients/Conv2D_24_grad/Conv2DBackpropInput.^gradients/Conv2D_24_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_24_grad/tuple/control_dependencyIdentity,gradients/Conv2D_24_grad/Conv2DBackpropInput*^gradients/Conv2D_24_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_24_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

3gradients/Conv2D_24_grad/tuple/control_dependency_1Identity-gradients/Conv2D_24_grad/Conv2DBackpropFilter*^gradients/Conv2D_24_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_24_grad/Conv2DBackpropFilter*&
_output_shapes
:>K
`
gradients/add_33_grad/ShapeShapePad_9*
T0*
out_type0*
_output_shapes
:
d
gradients/add_33_grad/Shape_1ShapeRelu_19*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_33_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_33_grad/Shapegradients/add_33_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ą
gradients/add_33_grad/SumSum1gradients/Conv2D_24_grad/tuple/control_dependency+gradients/add_33_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_33_grad/ReshapeReshapegradients/add_33_grad/Sumgradients/add_33_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
Ä
gradients/add_33_grad/Sum_1Sum1gradients/Conv2D_24_grad/tuple/control_dependency-gradients/add_33_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_33_grad/Reshape_1Reshapegradients/add_33_grad/Sum_1gradients/add_33_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
p
&gradients/add_33_grad/tuple/group_depsNoOp^gradients/add_33_grad/Reshape ^gradients/add_33_grad/Reshape_1

.gradients/add_33_grad/tuple/control_dependencyIdentitygradients/add_33_grad/Reshape'^gradients/add_33_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_33_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

0gradients/add_33_grad/tuple/control_dependency_1Identitygradients/add_33_grad/Reshape_1'^gradients/add_33_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_33_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
[
gradients/Pad_9_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
^
gradients/Pad_9_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 

gradients/Pad_9_grad/stackPackgradients/Pad_9_grad/Rankgradients/Pad_9_grad/stack/1*
N*
T0*

axis *
_output_shapes
:
q
 gradients/Pad_9_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
§
gradients/Pad_9_grad/SliceSlicePad_9/paddings gradients/Pad_9_grad/Slice/begingradients/Pad_9_grad/stack*
T0*
Index0*
_output_shapes

:
u
"gradients/Pad_9_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

gradients/Pad_9_grad/ReshapeReshapegradients/Pad_9_grad/Slice"gradients/Pad_9_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Pad_9_grad/ShapeShapeadd_30*
T0*
out_type0*
_output_shapes
:
č
gradients/Pad_9_grad/Slice_1Slice.gradients/add_33_grad/tuple/control_dependencygradients/Pad_9_grad/Reshapegradients/Pad_9_grad/Shape*
T0*
Index0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
²
gradients/Relu_19_grad/ReluGradReluGrad0gradients/add_33_grad/tuple/control_dependency_1Relu_19*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
d
gradients/add_32_grad/ShapeShape	Conv2D_23*
T0*
out_type0*
_output_shapes
:
g
gradients/add_32_grad/Shape_1Const*
valueB:>*
dtype0*
_output_shapes
:
½
+gradients/add_32_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_32_grad/Shapegradients/add_32_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_32_grad/SumSumgradients/Relu_19_grad/ReluGrad+gradients/add_32_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_32_grad/ReshapeReshapegradients/add_32_grad/Sumgradients/add_32_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
²
gradients/add_32_grad/Sum_1Sumgradients/Relu_19_grad/ReluGrad-gradients/add_32_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_32_grad/Reshape_1Reshapegradients/add_32_grad/Sum_1gradients/add_32_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:>
p
&gradients/add_32_grad/tuple/group_depsNoOp^gradients/add_32_grad/Reshape ^gradients/add_32_grad/Reshape_1

.gradients/add_32_grad/tuple/control_dependencyIdentitygradients/add_32_grad/Reshape'^gradients/add_32_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_32_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
ß
0gradients/add_32_grad/tuple/control_dependency_1Identitygradients/add_32_grad/Reshape_1'^gradients/add_32_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_32_grad/Reshape_1*
_output_shapes
:>
e
gradients/Conv2D_23_grad/ShapeShapeRelu_18*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_23_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_23_grad/Shape*Default/residual_block4/conv2/filters/read.gradients/add_32_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_23_grad/Shape_1Const*%
valueB"      >   >   *
dtype0*
_output_shapes
:
Ø
-gradients/Conv2D_23_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_18 gradients/Conv2D_23_grad/Shape_1.gradients/add_32_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:>>

)gradients/Conv2D_23_grad/tuple/group_depsNoOp-^gradients/Conv2D_23_grad/Conv2DBackpropInput.^gradients/Conv2D_23_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_23_grad/tuple/control_dependencyIdentity,gradients/Conv2D_23_grad/Conv2DBackpropInput*^gradients/Conv2D_23_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_23_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>

3gradients/Conv2D_23_grad/tuple/control_dependency_1Identity-gradients/Conv2D_23_grad/Conv2DBackpropFilter*^gradients/Conv2D_23_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_23_grad/Conv2DBackpropFilter*&
_output_shapes
:>>
³
gradients/Relu_18_grad/ReluGradReluGrad1gradients/Conv2D_23_grad/tuple/control_dependencyRelu_18*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
d
gradients/add_31_grad/ShapeShape	Conv2D_22*
T0*
out_type0*
_output_shapes
:
g
gradients/add_31_grad/Shape_1Const*
valueB:>*
dtype0*
_output_shapes
:
½
+gradients/add_31_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_31_grad/Shapegradients/add_31_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_31_grad/SumSumgradients/Relu_18_grad/ReluGrad+gradients/add_31_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_31_grad/ReshapeReshapegradients/add_31_grad/Sumgradients/add_31_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
²
gradients/add_31_grad/Sum_1Sumgradients/Relu_18_grad/ReluGrad-gradients/add_31_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_31_grad/Reshape_1Reshapegradients/add_31_grad/Sum_1gradients/add_31_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:>
p
&gradients/add_31_grad/tuple/group_depsNoOp^gradients/add_31_grad/Reshape ^gradients/add_31_grad/Reshape_1

.gradients/add_31_grad/tuple/control_dependencyIdentitygradients/add_31_grad/Reshape'^gradients/add_31_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_31_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’>
ß
0gradients/add_31_grad/tuple/control_dependency_1Identitygradients/add_31_grad/Reshape_1'^gradients/add_31_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_31_grad/Reshape_1*
_output_shapes
:>
d
gradients/Conv2D_22_grad/ShapeShapeadd_30*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_22_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_22_grad/Shape*Default/residual_block4/conv1/filters/read.gradients/add_31_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_22_grad/Shape_1Const*%
valueB"      4   >   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_22_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_30 gradients/Conv2D_22_grad/Shape_1.gradients/add_31_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:4>

)gradients/Conv2D_22_grad/tuple/group_depsNoOp-^gradients/Conv2D_22_grad/Conv2DBackpropInput.^gradients/Conv2D_22_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_22_grad/tuple/control_dependencyIdentity,gradients/Conv2D_22_grad/Conv2DBackpropInput*^gradients/Conv2D_22_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_22_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

3gradients/Conv2D_22_grad/tuple/control_dependency_1Identity-gradients/Conv2D_22_grad/Conv2DBackpropFilter*^gradients/Conv2D_22_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_22_grad/Conv2DBackpropFilter*&
_output_shapes
:4>
ļ
gradients/AddN_7AddNgradients/Pad_9_grad/Slice_11gradients/Conv2D_22_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/Pad_9_grad/Slice_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
`
gradients/add_30_grad/ShapeShapePad_8*
T0*
out_type0*
_output_shapes
:
d
gradients/add_30_grad/Shape_1ShapeRelu_17*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_30_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_30_grad/Shapegradients/add_30_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_30_grad/SumSumgradients/AddN_7+gradients/add_30_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_30_grad/ReshapeReshapegradients/add_30_grad/Sumgradients/add_30_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
£
gradients/add_30_grad/Sum_1Sumgradients/AddN_7-gradients/add_30_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_30_grad/Reshape_1Reshapegradients/add_30_grad/Sum_1gradients/add_30_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
p
&gradients/add_30_grad/tuple/group_depsNoOp^gradients/add_30_grad/Reshape ^gradients/add_30_grad/Reshape_1

.gradients/add_30_grad/tuple/control_dependencyIdentitygradients/add_30_grad/Reshape'^gradients/add_30_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_30_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

0gradients/add_30_grad/tuple/control_dependency_1Identitygradients/add_30_grad/Reshape_1'^gradients/add_30_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_30_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
[
gradients/Pad_8_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
^
gradients/Pad_8_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 

gradients/Pad_8_grad/stackPackgradients/Pad_8_grad/Rankgradients/Pad_8_grad/stack/1*
N*
T0*

axis *
_output_shapes
:
q
 gradients/Pad_8_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
§
gradients/Pad_8_grad/SliceSlicePad_8/paddings gradients/Pad_8_grad/Slice/begingradients/Pad_8_grad/stack*
T0*
Index0*
_output_shapes

:
u
"gradients/Pad_8_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

gradients/Pad_8_grad/ReshapeReshapegradients/Pad_8_grad/Slice"gradients/Pad_8_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Pad_8_grad/ShapeShapeadd_27*
T0*
out_type0*
_output_shapes
:
č
gradients/Pad_8_grad/Slice_1Slice.gradients/add_30_grad/tuple/control_dependencygradients/Pad_8_grad/Reshapegradients/Pad_8_grad/Shape*
T0*
Index0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
²
gradients/Relu_17_grad/ReluGradReluGrad0gradients/add_30_grad/tuple/control_dependency_1Relu_17*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
d
gradients/add_29_grad/ShapeShape	Conv2D_21*
T0*
out_type0*
_output_shapes
:
g
gradients/add_29_grad/Shape_1Const*
valueB:4*
dtype0*
_output_shapes
:
½
+gradients/add_29_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_29_grad/Shapegradients/add_29_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_29_grad/SumSumgradients/Relu_17_grad/ReluGrad+gradients/add_29_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_29_grad/ReshapeReshapegradients/add_29_grad/Sumgradients/add_29_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
²
gradients/add_29_grad/Sum_1Sumgradients/Relu_17_grad/ReluGrad-gradients/add_29_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_29_grad/Reshape_1Reshapegradients/add_29_grad/Sum_1gradients/add_29_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:4
p
&gradients/add_29_grad/tuple/group_depsNoOp^gradients/add_29_grad/Reshape ^gradients/add_29_grad/Reshape_1

.gradients/add_29_grad/tuple/control_dependencyIdentitygradients/add_29_grad/Reshape'^gradients/add_29_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_29_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
ß
0gradients/add_29_grad/tuple/control_dependency_1Identitygradients/add_29_grad/Reshape_1'^gradients/add_29_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_29_grad/Reshape_1*
_output_shapes
:4
e
gradients/Conv2D_21_grad/ShapeShapeRelu_16*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_21_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_21_grad/Shape*Default/residual_block3/conv2/filters/read.gradients/add_29_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_21_grad/Shape_1Const*%
valueB"      4   4   *
dtype0*
_output_shapes
:
Ø
-gradients/Conv2D_21_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_16 gradients/Conv2D_21_grad/Shape_1.gradients/add_29_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:44

)gradients/Conv2D_21_grad/tuple/group_depsNoOp-^gradients/Conv2D_21_grad/Conv2DBackpropInput.^gradients/Conv2D_21_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_21_grad/tuple/control_dependencyIdentity,gradients/Conv2D_21_grad/Conv2DBackpropInput*^gradients/Conv2D_21_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_21_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4

3gradients/Conv2D_21_grad/tuple/control_dependency_1Identity-gradients/Conv2D_21_grad/Conv2DBackpropFilter*^gradients/Conv2D_21_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_21_grad/Conv2DBackpropFilter*&
_output_shapes
:44
³
gradients/Relu_16_grad/ReluGradReluGrad1gradients/Conv2D_21_grad/tuple/control_dependencyRelu_16*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
d
gradients/add_28_grad/ShapeShape	Conv2D_20*
T0*
out_type0*
_output_shapes
:
g
gradients/add_28_grad/Shape_1Const*
valueB:4*
dtype0*
_output_shapes
:
½
+gradients/add_28_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_28_grad/Shapegradients/add_28_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_28_grad/SumSumgradients/Relu_16_grad/ReluGrad+gradients/add_28_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_28_grad/ReshapeReshapegradients/add_28_grad/Sumgradients/add_28_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
²
gradients/add_28_grad/Sum_1Sumgradients/Relu_16_grad/ReluGrad-gradients/add_28_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_28_grad/Reshape_1Reshapegradients/add_28_grad/Sum_1gradients/add_28_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:4
p
&gradients/add_28_grad/tuple/group_depsNoOp^gradients/add_28_grad/Reshape ^gradients/add_28_grad/Reshape_1

.gradients/add_28_grad/tuple/control_dependencyIdentitygradients/add_28_grad/Reshape'^gradients/add_28_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_28_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’4
ß
0gradients/add_28_grad/tuple/control_dependency_1Identitygradients/add_28_grad/Reshape_1'^gradients/add_28_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_28_grad/Reshape_1*
_output_shapes
:4
d
gradients/Conv2D_20_grad/ShapeShapeadd_27*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_20_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_20_grad/Shape*Default/residual_block3/conv1/filters/read.gradients/add_28_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_20_grad/Shape_1Const*%
valueB"      ,   4   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_20_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_27 gradients/Conv2D_20_grad/Shape_1.gradients/add_28_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:,4

)gradients/Conv2D_20_grad/tuple/group_depsNoOp-^gradients/Conv2D_20_grad/Conv2DBackpropInput.^gradients/Conv2D_20_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_20_grad/tuple/control_dependencyIdentity,gradients/Conv2D_20_grad/Conv2DBackpropInput*^gradients/Conv2D_20_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_20_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

3gradients/Conv2D_20_grad/tuple/control_dependency_1Identity-gradients/Conv2D_20_grad/Conv2DBackpropFilter*^gradients/Conv2D_20_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_20_grad/Conv2DBackpropFilter*&
_output_shapes
:,4
ļ
gradients/AddN_8AddNgradients/Pad_8_grad/Slice_11gradients/Conv2D_20_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/Pad_8_grad/Slice_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
`
gradients/add_27_grad/ShapeShapePad_7*
T0*
out_type0*
_output_shapes
:
d
gradients/add_27_grad/Shape_1ShapeRelu_15*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_27_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_27_grad/Shapegradients/add_27_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_27_grad/SumSumgradients/AddN_8+gradients/add_27_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_27_grad/ReshapeReshapegradients/add_27_grad/Sumgradients/add_27_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
£
gradients/add_27_grad/Sum_1Sumgradients/AddN_8-gradients/add_27_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_27_grad/Reshape_1Reshapegradients/add_27_grad/Sum_1gradients/add_27_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
p
&gradients/add_27_grad/tuple/group_depsNoOp^gradients/add_27_grad/Reshape ^gradients/add_27_grad/Reshape_1

.gradients/add_27_grad/tuple/control_dependencyIdentitygradients/add_27_grad/Reshape'^gradients/add_27_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_27_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

0gradients/add_27_grad/tuple/control_dependency_1Identitygradients/add_27_grad/Reshape_1'^gradients/add_27_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_27_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
[
gradients/Pad_7_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
^
gradients/Pad_7_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 

gradients/Pad_7_grad/stackPackgradients/Pad_7_grad/Rankgradients/Pad_7_grad/stack/1*
N*
T0*

axis *
_output_shapes
:
q
 gradients/Pad_7_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
§
gradients/Pad_7_grad/SliceSlicePad_7/paddings gradients/Pad_7_grad/Slice/begingradients/Pad_7_grad/stack*
T0*
Index0*
_output_shapes

:
u
"gradients/Pad_7_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

gradients/Pad_7_grad/ReshapeReshapegradients/Pad_7_grad/Slice"gradients/Pad_7_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Pad_7_grad/ShapeShapeadd_24*
T0*
out_type0*
_output_shapes
:
č
gradients/Pad_7_grad/Slice_1Slice.gradients/add_27_grad/tuple/control_dependencygradients/Pad_7_grad/Reshapegradients/Pad_7_grad/Shape*
T0*
Index0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
²
gradients/Relu_15_grad/ReluGradReluGrad0gradients/add_27_grad/tuple/control_dependency_1Relu_15*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
d
gradients/add_26_grad/ShapeShape	Conv2D_19*
T0*
out_type0*
_output_shapes
:
g
gradients/add_26_grad/Shape_1Const*
valueB:,*
dtype0*
_output_shapes
:
½
+gradients/add_26_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_26_grad/Shapegradients/add_26_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_26_grad/SumSumgradients/Relu_15_grad/ReluGrad+gradients/add_26_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_26_grad/ReshapeReshapegradients/add_26_grad/Sumgradients/add_26_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
²
gradients/add_26_grad/Sum_1Sumgradients/Relu_15_grad/ReluGrad-gradients/add_26_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_26_grad/Reshape_1Reshapegradients/add_26_grad/Sum_1gradients/add_26_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:,
p
&gradients/add_26_grad/tuple/group_depsNoOp^gradients/add_26_grad/Reshape ^gradients/add_26_grad/Reshape_1

.gradients/add_26_grad/tuple/control_dependencyIdentitygradients/add_26_grad/Reshape'^gradients/add_26_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_26_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
ß
0gradients/add_26_grad/tuple/control_dependency_1Identitygradients/add_26_grad/Reshape_1'^gradients/add_26_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_26_grad/Reshape_1*
_output_shapes
:,
e
gradients/Conv2D_19_grad/ShapeShapeRelu_14*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_19_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_19_grad/Shape*Default/residual_block2/conv2/filters/read.gradients/add_26_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_19_grad/Shape_1Const*%
valueB"      ,   ,   *
dtype0*
_output_shapes
:
Ø
-gradients/Conv2D_19_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_14 gradients/Conv2D_19_grad/Shape_1.gradients/add_26_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:,,

)gradients/Conv2D_19_grad/tuple/group_depsNoOp-^gradients/Conv2D_19_grad/Conv2DBackpropInput.^gradients/Conv2D_19_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_19_grad/tuple/control_dependencyIdentity,gradients/Conv2D_19_grad/Conv2DBackpropInput*^gradients/Conv2D_19_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_19_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,

3gradients/Conv2D_19_grad/tuple/control_dependency_1Identity-gradients/Conv2D_19_grad/Conv2DBackpropFilter*^gradients/Conv2D_19_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_19_grad/Conv2DBackpropFilter*&
_output_shapes
:,,
³
gradients/Relu_14_grad/ReluGradReluGrad1gradients/Conv2D_19_grad/tuple/control_dependencyRelu_14*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
d
gradients/add_25_grad/ShapeShape	Conv2D_18*
T0*
out_type0*
_output_shapes
:
g
gradients/add_25_grad/Shape_1Const*
valueB:,*
dtype0*
_output_shapes
:
½
+gradients/add_25_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_25_grad/Shapegradients/add_25_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_25_grad/SumSumgradients/Relu_14_grad/ReluGrad+gradients/add_25_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_25_grad/ReshapeReshapegradients/add_25_grad/Sumgradients/add_25_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
²
gradients/add_25_grad/Sum_1Sumgradients/Relu_14_grad/ReluGrad-gradients/add_25_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_25_grad/Reshape_1Reshapegradients/add_25_grad/Sum_1gradients/add_25_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:,
p
&gradients/add_25_grad/tuple/group_depsNoOp^gradients/add_25_grad/Reshape ^gradients/add_25_grad/Reshape_1

.gradients/add_25_grad/tuple/control_dependencyIdentitygradients/add_25_grad/Reshape'^gradients/add_25_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_25_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’,
ß
0gradients/add_25_grad/tuple/control_dependency_1Identitygradients/add_25_grad/Reshape_1'^gradients/add_25_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_25_grad/Reshape_1*
_output_shapes
:,
d
gradients/Conv2D_18_grad/ShapeShapeadd_24*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_18_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_18_grad/Shape*Default/residual_block2/conv1/filters/read.gradients/add_25_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_18_grad/Shape_1Const*%
valueB"      &   ,   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_18_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_24 gradients/Conv2D_18_grad/Shape_1.gradients/add_25_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:&,

)gradients/Conv2D_18_grad/tuple/group_depsNoOp-^gradients/Conv2D_18_grad/Conv2DBackpropInput.^gradients/Conv2D_18_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_18_grad/tuple/control_dependencyIdentity,gradients/Conv2D_18_grad/Conv2DBackpropInput*^gradients/Conv2D_18_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_18_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

3gradients/Conv2D_18_grad/tuple/control_dependency_1Identity-gradients/Conv2D_18_grad/Conv2DBackpropFilter*^gradients/Conv2D_18_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_18_grad/Conv2DBackpropFilter*&
_output_shapes
:&,
ļ
gradients/AddN_9AddNgradients/Pad_7_grad/Slice_11gradients/Conv2D_18_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/Pad_7_grad/Slice_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
`
gradients/add_24_grad/ShapeShapePad_6*
T0*
out_type0*
_output_shapes
:
d
gradients/add_24_grad/Shape_1ShapeRelu_13*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_24_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_24_grad/Shapegradients/add_24_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_24_grad/SumSumgradients/AddN_9+gradients/add_24_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_24_grad/ReshapeReshapegradients/add_24_grad/Sumgradients/add_24_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
£
gradients/add_24_grad/Sum_1Sumgradients/AddN_9-gradients/add_24_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_24_grad/Reshape_1Reshapegradients/add_24_grad/Sum_1gradients/add_24_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
p
&gradients/add_24_grad/tuple/group_depsNoOp^gradients/add_24_grad/Reshape ^gradients/add_24_grad/Reshape_1

.gradients/add_24_grad/tuple/control_dependencyIdentitygradients/add_24_grad/Reshape'^gradients/add_24_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_24_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

0gradients/add_24_grad/tuple/control_dependency_1Identitygradients/add_24_grad/Reshape_1'^gradients/add_24_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_24_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
[
gradients/Pad_6_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
^
gradients/Pad_6_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 

gradients/Pad_6_grad/stackPackgradients/Pad_6_grad/Rankgradients/Pad_6_grad/stack/1*
N*
T0*

axis *
_output_shapes
:
q
 gradients/Pad_6_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
§
gradients/Pad_6_grad/SliceSlicePad_6/paddings gradients/Pad_6_grad/Slice/begingradients/Pad_6_grad/stack*
T0*
Index0*
_output_shapes

:
u
"gradients/Pad_6_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

gradients/Pad_6_grad/ReshapeReshapegradients/Pad_6_grad/Slice"gradients/Pad_6_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Pad_6_grad/ShapeShapeadd_21*
T0*
out_type0*
_output_shapes
:
č
gradients/Pad_6_grad/Slice_1Slice.gradients/add_24_grad/tuple/control_dependencygradients/Pad_6_grad/Reshapegradients/Pad_6_grad/Shape*
T0*
Index0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
²
gradients/Relu_13_grad/ReluGradReluGrad0gradients/add_24_grad/tuple/control_dependency_1Relu_13*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
d
gradients/add_23_grad/ShapeShape	Conv2D_17*
T0*
out_type0*
_output_shapes
:
g
gradients/add_23_grad/Shape_1Const*
valueB:&*
dtype0*
_output_shapes
:
½
+gradients/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_23_grad/Shapegradients/add_23_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_23_grad/SumSumgradients/Relu_13_grad/ReluGrad+gradients/add_23_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_23_grad/ReshapeReshapegradients/add_23_grad/Sumgradients/add_23_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
²
gradients/add_23_grad/Sum_1Sumgradients/Relu_13_grad/ReluGrad-gradients/add_23_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_23_grad/Reshape_1Reshapegradients/add_23_grad/Sum_1gradients/add_23_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:&
p
&gradients/add_23_grad/tuple/group_depsNoOp^gradients/add_23_grad/Reshape ^gradients/add_23_grad/Reshape_1

.gradients/add_23_grad/tuple/control_dependencyIdentitygradients/add_23_grad/Reshape'^gradients/add_23_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_23_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
ß
0gradients/add_23_grad/tuple/control_dependency_1Identitygradients/add_23_grad/Reshape_1'^gradients/add_23_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_23_grad/Reshape_1*
_output_shapes
:&
e
gradients/Conv2D_17_grad/ShapeShapeRelu_12*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_17_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_17_grad/Shape*Default/residual_block1/conv2/filters/read.gradients/add_23_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_17_grad/Shape_1Const*%
valueB"      &   &   *
dtype0*
_output_shapes
:
Ø
-gradients/Conv2D_17_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_12 gradients/Conv2D_17_grad/Shape_1.gradients/add_23_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:&&

)gradients/Conv2D_17_grad/tuple/group_depsNoOp-^gradients/Conv2D_17_grad/Conv2DBackpropInput.^gradients/Conv2D_17_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_17_grad/tuple/control_dependencyIdentity,gradients/Conv2D_17_grad/Conv2DBackpropInput*^gradients/Conv2D_17_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_17_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&

3gradients/Conv2D_17_grad/tuple/control_dependency_1Identity-gradients/Conv2D_17_grad/Conv2DBackpropFilter*^gradients/Conv2D_17_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_17_grad/Conv2DBackpropFilter*&
_output_shapes
:&&
³
gradients/Relu_12_grad/ReluGradReluGrad1gradients/Conv2D_17_grad/tuple/control_dependencyRelu_12*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
d
gradients/add_22_grad/ShapeShape	Conv2D_16*
T0*
out_type0*
_output_shapes
:
g
gradients/add_22_grad/Shape_1Const*
valueB:&*
dtype0*
_output_shapes
:
½
+gradients/add_22_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_22_grad/Shapegradients/add_22_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_22_grad/SumSumgradients/Relu_12_grad/ReluGrad+gradients/add_22_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_22_grad/ReshapeReshapegradients/add_22_grad/Sumgradients/add_22_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
²
gradients/add_22_grad/Sum_1Sumgradients/Relu_12_grad/ReluGrad-gradients/add_22_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_22_grad/Reshape_1Reshapegradients/add_22_grad/Sum_1gradients/add_22_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:&
p
&gradients/add_22_grad/tuple/group_depsNoOp^gradients/add_22_grad/Reshape ^gradients/add_22_grad/Reshape_1

.gradients/add_22_grad/tuple/control_dependencyIdentitygradients/add_22_grad/Reshape'^gradients/add_22_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_22_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’&
ß
0gradients/add_22_grad/tuple/control_dependency_1Identitygradients/add_22_grad/Reshape_1'^gradients/add_22_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_22_grad/Reshape_1*
_output_shapes
:&
d
gradients/Conv2D_16_grad/ShapeShapeadd_21*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_16_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_16_grad/Shape*Default/residual_block1/conv1/filters/read.gradients/add_22_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_16_grad/Shape_1Const*%
valueB"      "   &   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_16_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_21 gradients/Conv2D_16_grad/Shape_1.gradients/add_22_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:"&

)gradients/Conv2D_16_grad/tuple/group_depsNoOp-^gradients/Conv2D_16_grad/Conv2DBackpropInput.^gradients/Conv2D_16_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_16_grad/tuple/control_dependencyIdentity,gradients/Conv2D_16_grad/Conv2DBackpropInput*^gradients/Conv2D_16_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_16_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

3gradients/Conv2D_16_grad/tuple/control_dependency_1Identity-gradients/Conv2D_16_grad/Conv2DBackpropFilter*^gradients/Conv2D_16_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_16_grad/Conv2DBackpropFilter*&
_output_shapes
:"&
š
gradients/AddN_10AddNgradients/Pad_6_grad/Slice_11gradients/Conv2D_16_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/Pad_6_grad/Slice_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
`
gradients/add_21_grad/ShapeShapePad_5*
T0*
out_type0*
_output_shapes
:
d
gradients/add_21_grad/Shape_1ShapeRelu_11*
T0*
out_type0*
_output_shapes
:
½
+gradients/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_21_grad/Shapegradients/add_21_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
 
gradients/add_21_grad/SumSumgradients/AddN_10+gradients/add_21_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_21_grad/ReshapeReshapegradients/add_21_grad/Sumgradients/add_21_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
¤
gradients/add_21_grad/Sum_1Sumgradients/AddN_10-gradients/add_21_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ą
gradients/add_21_grad/Reshape_1Reshapegradients/add_21_grad/Sum_1gradients/add_21_grad/Shape_1*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
p
&gradients/add_21_grad/tuple/group_depsNoOp^gradients/add_21_grad/Reshape ^gradients/add_21_grad/Reshape_1

.gradients/add_21_grad/tuple/control_dependencyIdentitygradients/add_21_grad/Reshape'^gradients/add_21_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_21_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

0gradients/add_21_grad/tuple/control_dependency_1Identitygradients/add_21_grad/Reshape_1'^gradients/add_21_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_21_grad/Reshape_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
[
gradients/Pad_5_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
^
gradients/Pad_5_grad/stack/1Const*
value	B :*
dtype0*
_output_shapes
: 

gradients/Pad_5_grad/stackPackgradients/Pad_5_grad/Rankgradients/Pad_5_grad/stack/1*
N*
T0*

axis *
_output_shapes
:
q
 gradients/Pad_5_grad/Slice/beginConst*
valueB"        *
dtype0*
_output_shapes
:
§
gradients/Pad_5_grad/SliceSlicePad_5/paddings gradients/Pad_5_grad/Slice/begingradients/Pad_5_grad/stack*
T0*
Index0*
_output_shapes

:
u
"gradients/Pad_5_grad/Reshape/shapeConst*
valueB:
’’’’’’’’’*
dtype0*
_output_shapes
:

gradients/Pad_5_grad/ReshapeReshapegradients/Pad_5_grad/Slice"gradients/Pad_5_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Pad_5_grad/ShapeShapeadd_18*
T0*
out_type0*
_output_shapes
:
č
gradients/Pad_5_grad/Slice_1Slice.gradients/add_21_grad/tuple/control_dependencygradients/Pad_5_grad/Reshapegradients/Pad_5_grad/Shape*
T0*
Index0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
²
gradients/Relu_11_grad/ReluGradReluGrad0gradients/add_21_grad/tuple/control_dependency_1Relu_11*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
d
gradients/add_20_grad/ShapeShape	Conv2D_15*
T0*
out_type0*
_output_shapes
:
g
gradients/add_20_grad/Shape_1Const*
valueB:"*
dtype0*
_output_shapes
:
½
+gradients/add_20_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_20_grad/Shapegradients/add_20_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_20_grad/SumSumgradients/Relu_11_grad/ReluGrad+gradients/add_20_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_20_grad/ReshapeReshapegradients/add_20_grad/Sumgradients/add_20_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
²
gradients/add_20_grad/Sum_1Sumgradients/Relu_11_grad/ReluGrad-gradients/add_20_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_20_grad/Reshape_1Reshapegradients/add_20_grad/Sum_1gradients/add_20_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:"
p
&gradients/add_20_grad/tuple/group_depsNoOp^gradients/add_20_grad/Reshape ^gradients/add_20_grad/Reshape_1

.gradients/add_20_grad/tuple/control_dependencyIdentitygradients/add_20_grad/Reshape'^gradients/add_20_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_20_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
ß
0gradients/add_20_grad/tuple/control_dependency_1Identitygradients/add_20_grad/Reshape_1'^gradients/add_20_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_20_grad/Reshape_1*
_output_shapes
:"
e
gradients/Conv2D_15_grad/ShapeShapeRelu_10*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_15_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_15_grad/Shape*Default/residual_block0/conv2/filters/read.gradients/add_20_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_15_grad/Shape_1Const*%
valueB"      "   "   *
dtype0*
_output_shapes
:
Ø
-gradients/Conv2D_15_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu_10 gradients/Conv2D_15_grad/Shape_1.gradients/add_20_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:""

)gradients/Conv2D_15_grad/tuple/group_depsNoOp-^gradients/Conv2D_15_grad/Conv2DBackpropInput.^gradients/Conv2D_15_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_15_grad/tuple/control_dependencyIdentity,gradients/Conv2D_15_grad/Conv2DBackpropInput*^gradients/Conv2D_15_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_15_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"

3gradients/Conv2D_15_grad/tuple/control_dependency_1Identity-gradients/Conv2D_15_grad/Conv2DBackpropFilter*^gradients/Conv2D_15_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_15_grad/Conv2DBackpropFilter*&
_output_shapes
:""
³
gradients/Relu_10_grad/ReluGradReluGrad1gradients/Conv2D_15_grad/tuple/control_dependencyRelu_10*
T0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
d
gradients/add_19_grad/ShapeShape	Conv2D_14*
T0*
out_type0*
_output_shapes
:
g
gradients/add_19_grad/Shape_1Const*
valueB:"*
dtype0*
_output_shapes
:
½
+gradients/add_19_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_19_grad/Shapegradients/add_19_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
®
gradients/add_19_grad/SumSumgradients/Relu_10_grad/ReluGrad+gradients/add_19_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_19_grad/ReshapeReshapegradients/add_19_grad/Sumgradients/add_19_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
²
gradients/add_19_grad/Sum_1Sumgradients/Relu_10_grad/ReluGrad-gradients/add_19_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_19_grad/Reshape_1Reshapegradients/add_19_grad/Sum_1gradients/add_19_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:"
p
&gradients/add_19_grad/tuple/group_depsNoOp^gradients/add_19_grad/Reshape ^gradients/add_19_grad/Reshape_1

.gradients/add_19_grad/tuple/control_dependencyIdentitygradients/add_19_grad/Reshape'^gradients/add_19_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_19_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’"
ß
0gradients/add_19_grad/tuple/control_dependency_1Identitygradients/add_19_grad/Reshape_1'^gradients/add_19_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_19_grad/Reshape_1*
_output_shapes
:"
d
gradients/Conv2D_14_grad/ShapeShapeadd_18*
T0*
out_type0*
_output_shapes
:
ė
,gradients/Conv2D_14_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_14_grad/Shape*Default/residual_block0/conv1/filters/read.gradients/add_19_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_14_grad/Shape_1Const*%
valueB"          "   *
dtype0*
_output_shapes
:
§
-gradients/Conv2D_14_grad/Conv2DBackpropFilterConv2DBackpropFilteradd_18 gradients/Conv2D_14_grad/Shape_1.gradients/add_19_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: "

)gradients/Conv2D_14_grad/tuple/group_depsNoOp-^gradients/Conv2D_14_grad/Conv2DBackpropInput.^gradients/Conv2D_14_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_14_grad/tuple/control_dependencyIdentity,gradients/Conv2D_14_grad/Conv2DBackpropInput*^gradients/Conv2D_14_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_14_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 

3gradients/Conv2D_14_grad/tuple/control_dependency_1Identity-gradients/Conv2D_14_grad/Conv2DBackpropFilter*^gradients/Conv2D_14_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_14_grad/Conv2DBackpropFilter*&
_output_shapes
: "
š
gradients/AddN_11AddNgradients/Pad_5_grad/Slice_11gradients/Conv2D_14_grad/tuple/control_dependency*
N*
T0*/
_class%
#!loc:@gradients/Pad_5_grad/Slice_1*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
d
gradients/add_18_grad/ShapeShape	Conv2D_13*
T0*
out_type0*
_output_shapes
:
g
gradients/add_18_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
½
+gradients/add_18_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_18_grad/Shapegradients/add_18_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
 
gradients/add_18_grad/SumSumgradients/AddN_11+gradients/add_18_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
ŗ
gradients/add_18_grad/ReshapeReshapegradients/add_18_grad/Sumgradients/add_18_grad/Shape*
T0*
Tshape0*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
¤
gradients/add_18_grad/Sum_1Sumgradients/AddN_11-gradients/add_18_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:

gradients/add_18_grad/Reshape_1Reshapegradients/add_18_grad/Sum_1gradients/add_18_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
p
&gradients/add_18_grad/tuple/group_depsNoOp^gradients/add_18_grad/Reshape ^gradients/add_18_grad/Reshape_1

.gradients/add_18_grad/tuple/control_dependencyIdentitygradients/add_18_grad/Reshape'^gradients/add_18_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/add_18_grad/Reshape*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’ 
ß
0gradients/add_18_grad/tuple/control_dependency_1Identitygradients/add_18_grad/Reshape_1'^gradients/add_18_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_18_grad/Reshape_1*
_output_shapes
: 
k
gradients/Conv2D_13_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
į
,gradients/Conv2D_13_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_13_grad/Shape Default/first_layer/filters/read.gradients/add_18_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
y
 gradients/Conv2D_13_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:
®
-gradients/Conv2D_13_grad/Conv2DBackpropFilterConv2DBackpropFilterPlaceholder_3 gradients/Conv2D_13_grad/Shape_1.gradients/add_18_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: 

)gradients/Conv2D_13_grad/tuple/group_depsNoOp-^gradients/Conv2D_13_grad/Conv2DBackpropInput.^gradients/Conv2D_13_grad/Conv2DBackpropFilter
¤
1gradients/Conv2D_13_grad/tuple/control_dependencyIdentity,gradients/Conv2D_13_grad/Conv2DBackpropInput*^gradients/Conv2D_13_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_13_grad/Conv2DBackpropInput*A
_output_shapes/
-:+’’’’’’’’’’’’’’’’’’’’’’’’’’’

3gradients/Conv2D_13_grad/tuple/control_dependency_1Identity-gradients/Conv2D_13_grad/Conv2DBackpropFilter*^gradients/Conv2D_13_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/Conv2D_13_grad/Conv2DBackpropFilter*&
_output_shapes
: 

beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
¾
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
z
beta1_power/readIdentitybeta1_power*
T0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

beta2_power/initial_valueConst*
valueB
 *w¾?*
dtype0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
¾
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
z
beta2_power/readIdentitybeta2_power*
T0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
Ē
2Default/first_layer/filters/Adam/Initializer/zerosConst*%
valueB *    *
dtype0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ō
 Default/first_layer/filters/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

'Default/first_layer/filters/Adam/AssignAssign Default/first_layer/filters/Adam2Default/first_layer/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
“
%Default/first_layer/filters/Adam/readIdentity Default/first_layer/filters/Adam*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
É
4Default/first_layer/filters/Adam_1/Initializer/zerosConst*%
valueB *    *
dtype0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ö
"Default/first_layer/filters/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

)Default/first_layer/filters/Adam_1/AssignAssign"Default/first_layer/filters/Adam_14Default/first_layer/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
ø
'Default/first_layer/filters/Adam_1/readIdentity"Default/first_layer/filters/Adam_1*
T0*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
­
1Default/first_layer/biases/Adam/Initializer/zerosConst*
valueB *    *
dtype0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
ŗ
Default/first_layer/biases/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

&Default/first_layer/biases/Adam/AssignAssignDefault/first_layer/biases/Adam1Default/first_layer/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
„
$Default/first_layer/biases/Adam/readIdentityDefault/first_layer/biases/Adam*
T0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
Æ
3Default/first_layer/biases/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
¼
!Default/first_layer/biases/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

(Default/first_layer/biases/Adam_1/AssignAssign!Default/first_layer/biases/Adam_13Default/first_layer/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
©
&Default/first_layer/biases/Adam_1/readIdentity!Default/first_layer/biases/Adam_1*
T0*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
Ū
<Default/residual_block0/conv1/filters/Adam/Initializer/zerosConst*%
valueB "*    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
č
*Default/residual_block0/conv1/filters/Adam
VariableV2*
shape: "*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
¹
1Default/residual_block0/conv1/filters/Adam/AssignAssign*Default/residual_block0/conv1/filters/Adam<Default/residual_block0/conv1/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Ņ
/Default/residual_block0/conv1/filters/Adam/readIdentity*Default/residual_block0/conv1/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Ż
>Default/residual_block0/conv1/filters/Adam_1/Initializer/zerosConst*%
valueB "*    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
ź
,Default/residual_block0/conv1/filters/Adam_1
VariableV2*
shape: "*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
æ
3Default/residual_block0/conv1/filters/Adam_1/AssignAssign,Default/residual_block0/conv1/filters/Adam_1>Default/residual_block0/conv1/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Ö
1Default/residual_block0/conv1/filters/Adam_1/readIdentity,Default/residual_block0/conv1/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Į
;Default/residual_block0/conv1/biases/Adam/Initializer/zerosConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Ī
)Default/residual_block0/conv1/biases/Adam
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
©
0Default/residual_block0/conv1/biases/Adam/AssignAssign)Default/residual_block0/conv1/biases/Adam;Default/residual_block0/conv1/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Ć
.Default/residual_block0/conv1/biases/Adam/readIdentity)Default/residual_block0/conv1/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Ć
=Default/residual_block0/conv1/biases/Adam_1/Initializer/zerosConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Š
+Default/residual_block0/conv1/biases/Adam_1
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Æ
2Default/residual_block0/conv1/biases/Adam_1/AssignAssign+Default/residual_block0/conv1/biases/Adam_1=Default/residual_block0/conv1/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Ē
0Default/residual_block0/conv1/biases/Adam_1/readIdentity+Default/residual_block0/conv1/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
Ū
<Default/residual_block0/conv2/filters/Adam/Initializer/zerosConst*%
valueB""*    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
č
*Default/residual_block0/conv2/filters/Adam
VariableV2*
shape:""*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
¹
1Default/residual_block0/conv2/filters/Adam/AssignAssign*Default/residual_block0/conv2/filters/Adam<Default/residual_block0/conv2/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Ņ
/Default/residual_block0/conv2/filters/Adam/readIdentity*Default/residual_block0/conv2/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Ż
>Default/residual_block0/conv2/filters/Adam_1/Initializer/zerosConst*%
valueB""*    *
dtype0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
ź
,Default/residual_block0/conv2/filters/Adam_1
VariableV2*
shape:""*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
æ
3Default/residual_block0/conv2/filters/Adam_1/AssignAssign,Default/residual_block0/conv2/filters/Adam_1>Default/residual_block0/conv2/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Ö
1Default/residual_block0/conv2/filters/Adam_1/readIdentity,Default/residual_block0/conv2/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Į
;Default/residual_block0/conv2/biases/Adam/Initializer/zerosConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Ī
)Default/residual_block0/conv2/biases/Adam
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
©
0Default/residual_block0/conv2/biases/Adam/AssignAssign)Default/residual_block0/conv2/biases/Adam;Default/residual_block0/conv2/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Ć
.Default/residual_block0/conv2/biases/Adam/readIdentity)Default/residual_block0/conv2/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Ć
=Default/residual_block0/conv2/biases/Adam_1/Initializer/zerosConst*
valueB"*    *
dtype0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Š
+Default/residual_block0/conv2/biases/Adam_1
VariableV2*
shape:"*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Æ
2Default/residual_block0/conv2/biases/Adam_1/AssignAssign+Default/residual_block0/conv2/biases/Adam_1=Default/residual_block0/conv2/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Ē
0Default/residual_block0/conv2/biases/Adam_1/readIdentity+Default/residual_block0/conv2/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
Ū
<Default/residual_block1/conv1/filters/Adam/Initializer/zerosConst*%
valueB"&*    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
č
*Default/residual_block1/conv1/filters/Adam
VariableV2*
shape:"&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
¹
1Default/residual_block1/conv1/filters/Adam/AssignAssign*Default/residual_block1/conv1/filters/Adam<Default/residual_block1/conv1/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Ņ
/Default/residual_block1/conv1/filters/Adam/readIdentity*Default/residual_block1/conv1/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Ż
>Default/residual_block1/conv1/filters/Adam_1/Initializer/zerosConst*%
valueB"&*    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
ź
,Default/residual_block1/conv1/filters/Adam_1
VariableV2*
shape:"&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
æ
3Default/residual_block1/conv1/filters/Adam_1/AssignAssign,Default/residual_block1/conv1/filters/Adam_1>Default/residual_block1/conv1/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Ö
1Default/residual_block1/conv1/filters/Adam_1/readIdentity,Default/residual_block1/conv1/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Į
;Default/residual_block1/conv1/biases/Adam/Initializer/zerosConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Ī
)Default/residual_block1/conv1/biases/Adam
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
©
0Default/residual_block1/conv1/biases/Adam/AssignAssign)Default/residual_block1/conv1/biases/Adam;Default/residual_block1/conv1/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Ć
.Default/residual_block1/conv1/biases/Adam/readIdentity)Default/residual_block1/conv1/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Ć
=Default/residual_block1/conv1/biases/Adam_1/Initializer/zerosConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Š
+Default/residual_block1/conv1/biases/Adam_1
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Æ
2Default/residual_block1/conv1/biases/Adam_1/AssignAssign+Default/residual_block1/conv1/biases/Adam_1=Default/residual_block1/conv1/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Ē
0Default/residual_block1/conv1/biases/Adam_1/readIdentity+Default/residual_block1/conv1/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
Ū
<Default/residual_block1/conv2/filters/Adam/Initializer/zerosConst*%
valueB&&*    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
č
*Default/residual_block1/conv2/filters/Adam
VariableV2*
shape:&&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
¹
1Default/residual_block1/conv2/filters/Adam/AssignAssign*Default/residual_block1/conv2/filters/Adam<Default/residual_block1/conv2/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Ņ
/Default/residual_block1/conv2/filters/Adam/readIdentity*Default/residual_block1/conv2/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Ż
>Default/residual_block1/conv2/filters/Adam_1/Initializer/zerosConst*%
valueB&&*    *
dtype0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
ź
,Default/residual_block1/conv2/filters/Adam_1
VariableV2*
shape:&&*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
æ
3Default/residual_block1/conv2/filters/Adam_1/AssignAssign,Default/residual_block1/conv2/filters/Adam_1>Default/residual_block1/conv2/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Ö
1Default/residual_block1/conv2/filters/Adam_1/readIdentity,Default/residual_block1/conv2/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Į
;Default/residual_block1/conv2/biases/Adam/Initializer/zerosConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Ī
)Default/residual_block1/conv2/biases/Adam
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
©
0Default/residual_block1/conv2/biases/Adam/AssignAssign)Default/residual_block1/conv2/biases/Adam;Default/residual_block1/conv2/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Ć
.Default/residual_block1/conv2/biases/Adam/readIdentity)Default/residual_block1/conv2/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Ć
=Default/residual_block1/conv2/biases/Adam_1/Initializer/zerosConst*
valueB&*    *
dtype0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Š
+Default/residual_block1/conv2/biases/Adam_1
VariableV2*
shape:&*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Æ
2Default/residual_block1/conv2/biases/Adam_1/AssignAssign+Default/residual_block1/conv2/biases/Adam_1=Default/residual_block1/conv2/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Ē
0Default/residual_block1/conv2/biases/Adam_1/readIdentity+Default/residual_block1/conv2/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
Ū
<Default/residual_block2/conv1/filters/Adam/Initializer/zerosConst*%
valueB&,*    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
č
*Default/residual_block2/conv1/filters/Adam
VariableV2*
shape:&,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
¹
1Default/residual_block2/conv1/filters/Adam/AssignAssign*Default/residual_block2/conv1/filters/Adam<Default/residual_block2/conv1/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Ņ
/Default/residual_block2/conv1/filters/Adam/readIdentity*Default/residual_block2/conv1/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Ż
>Default/residual_block2/conv1/filters/Adam_1/Initializer/zerosConst*%
valueB&,*    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
ź
,Default/residual_block2/conv1/filters/Adam_1
VariableV2*
shape:&,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
æ
3Default/residual_block2/conv1/filters/Adam_1/AssignAssign,Default/residual_block2/conv1/filters/Adam_1>Default/residual_block2/conv1/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Ö
1Default/residual_block2/conv1/filters/Adam_1/readIdentity,Default/residual_block2/conv1/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Į
;Default/residual_block2/conv1/biases/Adam/Initializer/zerosConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Ī
)Default/residual_block2/conv1/biases/Adam
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
©
0Default/residual_block2/conv1/biases/Adam/AssignAssign)Default/residual_block2/conv1/biases/Adam;Default/residual_block2/conv1/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Ć
.Default/residual_block2/conv1/biases/Adam/readIdentity)Default/residual_block2/conv1/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Ć
=Default/residual_block2/conv1/biases/Adam_1/Initializer/zerosConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Š
+Default/residual_block2/conv1/biases/Adam_1
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Æ
2Default/residual_block2/conv1/biases/Adam_1/AssignAssign+Default/residual_block2/conv1/biases/Adam_1=Default/residual_block2/conv1/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Ē
0Default/residual_block2/conv1/biases/Adam_1/readIdentity+Default/residual_block2/conv1/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
Ū
<Default/residual_block2/conv2/filters/Adam/Initializer/zerosConst*%
valueB,,*    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
č
*Default/residual_block2/conv2/filters/Adam
VariableV2*
shape:,,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
¹
1Default/residual_block2/conv2/filters/Adam/AssignAssign*Default/residual_block2/conv2/filters/Adam<Default/residual_block2/conv2/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Ņ
/Default/residual_block2/conv2/filters/Adam/readIdentity*Default/residual_block2/conv2/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Ż
>Default/residual_block2/conv2/filters/Adam_1/Initializer/zerosConst*%
valueB,,*    *
dtype0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
ź
,Default/residual_block2/conv2/filters/Adam_1
VariableV2*
shape:,,*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
æ
3Default/residual_block2/conv2/filters/Adam_1/AssignAssign,Default/residual_block2/conv2/filters/Adam_1>Default/residual_block2/conv2/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Ö
1Default/residual_block2/conv2/filters/Adam_1/readIdentity,Default/residual_block2/conv2/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Į
;Default/residual_block2/conv2/biases/Adam/Initializer/zerosConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Ī
)Default/residual_block2/conv2/biases/Adam
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
©
0Default/residual_block2/conv2/biases/Adam/AssignAssign)Default/residual_block2/conv2/biases/Adam;Default/residual_block2/conv2/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Ć
.Default/residual_block2/conv2/biases/Adam/readIdentity)Default/residual_block2/conv2/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Ć
=Default/residual_block2/conv2/biases/Adam_1/Initializer/zerosConst*
valueB,*    *
dtype0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Š
+Default/residual_block2/conv2/biases/Adam_1
VariableV2*
shape:,*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Æ
2Default/residual_block2/conv2/biases/Adam_1/AssignAssign+Default/residual_block2/conv2/biases/Adam_1=Default/residual_block2/conv2/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Ē
0Default/residual_block2/conv2/biases/Adam_1/readIdentity+Default/residual_block2/conv2/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
Ū
<Default/residual_block3/conv1/filters/Adam/Initializer/zerosConst*%
valueB,4*    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
č
*Default/residual_block3/conv1/filters/Adam
VariableV2*
shape:,4*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
¹
1Default/residual_block3/conv1/filters/Adam/AssignAssign*Default/residual_block3/conv1/filters/Adam<Default/residual_block3/conv1/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Ņ
/Default/residual_block3/conv1/filters/Adam/readIdentity*Default/residual_block3/conv1/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Ż
>Default/residual_block3/conv1/filters/Adam_1/Initializer/zerosConst*%
valueB,4*    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
ź
,Default/residual_block3/conv1/filters/Adam_1
VariableV2*
shape:,4*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
æ
3Default/residual_block3/conv1/filters/Adam_1/AssignAssign,Default/residual_block3/conv1/filters/Adam_1>Default/residual_block3/conv1/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Ö
1Default/residual_block3/conv1/filters/Adam_1/readIdentity,Default/residual_block3/conv1/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Į
;Default/residual_block3/conv1/biases/Adam/Initializer/zerosConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Ī
)Default/residual_block3/conv1/biases/Adam
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
©
0Default/residual_block3/conv1/biases/Adam/AssignAssign)Default/residual_block3/conv1/biases/Adam;Default/residual_block3/conv1/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Ć
.Default/residual_block3/conv1/biases/Adam/readIdentity)Default/residual_block3/conv1/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Ć
=Default/residual_block3/conv1/biases/Adam_1/Initializer/zerosConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Š
+Default/residual_block3/conv1/biases/Adam_1
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Æ
2Default/residual_block3/conv1/biases/Adam_1/AssignAssign+Default/residual_block3/conv1/biases/Adam_1=Default/residual_block3/conv1/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Ē
0Default/residual_block3/conv1/biases/Adam_1/readIdentity+Default/residual_block3/conv1/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
Ū
<Default/residual_block3/conv2/filters/Adam/Initializer/zerosConst*%
valueB44*    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
č
*Default/residual_block3/conv2/filters/Adam
VariableV2*
shape:44*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
¹
1Default/residual_block3/conv2/filters/Adam/AssignAssign*Default/residual_block3/conv2/filters/Adam<Default/residual_block3/conv2/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Ņ
/Default/residual_block3/conv2/filters/Adam/readIdentity*Default/residual_block3/conv2/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Ż
>Default/residual_block3/conv2/filters/Adam_1/Initializer/zerosConst*%
valueB44*    *
dtype0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
ź
,Default/residual_block3/conv2/filters/Adam_1
VariableV2*
shape:44*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
æ
3Default/residual_block3/conv2/filters/Adam_1/AssignAssign,Default/residual_block3/conv2/filters/Adam_1>Default/residual_block3/conv2/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Ö
1Default/residual_block3/conv2/filters/Adam_1/readIdentity,Default/residual_block3/conv2/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Į
;Default/residual_block3/conv2/biases/Adam/Initializer/zerosConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Ī
)Default/residual_block3/conv2/biases/Adam
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
©
0Default/residual_block3/conv2/biases/Adam/AssignAssign)Default/residual_block3/conv2/biases/Adam;Default/residual_block3/conv2/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Ć
.Default/residual_block3/conv2/biases/Adam/readIdentity)Default/residual_block3/conv2/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Ć
=Default/residual_block3/conv2/biases/Adam_1/Initializer/zerosConst*
valueB4*    *
dtype0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Š
+Default/residual_block3/conv2/biases/Adam_1
VariableV2*
shape:4*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Æ
2Default/residual_block3/conv2/biases/Adam_1/AssignAssign+Default/residual_block3/conv2/biases/Adam_1=Default/residual_block3/conv2/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Ē
0Default/residual_block3/conv2/biases/Adam_1/readIdentity+Default/residual_block3/conv2/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
Ū
<Default/residual_block4/conv1/filters/Adam/Initializer/zerosConst*%
valueB4>*    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
č
*Default/residual_block4/conv1/filters/Adam
VariableV2*
shape:4>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
¹
1Default/residual_block4/conv1/filters/Adam/AssignAssign*Default/residual_block4/conv1/filters/Adam<Default/residual_block4/conv1/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Ņ
/Default/residual_block4/conv1/filters/Adam/readIdentity*Default/residual_block4/conv1/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Ż
>Default/residual_block4/conv1/filters/Adam_1/Initializer/zerosConst*%
valueB4>*    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
ź
,Default/residual_block4/conv1/filters/Adam_1
VariableV2*
shape:4>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
æ
3Default/residual_block4/conv1/filters/Adam_1/AssignAssign,Default/residual_block4/conv1/filters/Adam_1>Default/residual_block4/conv1/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Ö
1Default/residual_block4/conv1/filters/Adam_1/readIdentity,Default/residual_block4/conv1/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Į
;Default/residual_block4/conv1/biases/Adam/Initializer/zerosConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Ī
)Default/residual_block4/conv1/biases/Adam
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
©
0Default/residual_block4/conv1/biases/Adam/AssignAssign)Default/residual_block4/conv1/biases/Adam;Default/residual_block4/conv1/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Ć
.Default/residual_block4/conv1/biases/Adam/readIdentity)Default/residual_block4/conv1/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Ć
=Default/residual_block4/conv1/biases/Adam_1/Initializer/zerosConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Š
+Default/residual_block4/conv1/biases/Adam_1
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Æ
2Default/residual_block4/conv1/biases/Adam_1/AssignAssign+Default/residual_block4/conv1/biases/Adam_1=Default/residual_block4/conv1/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Ē
0Default/residual_block4/conv1/biases/Adam_1/readIdentity+Default/residual_block4/conv1/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
Ū
<Default/residual_block4/conv2/filters/Adam/Initializer/zerosConst*%
valueB>>*    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
č
*Default/residual_block4/conv2/filters/Adam
VariableV2*
shape:>>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
¹
1Default/residual_block4/conv2/filters/Adam/AssignAssign*Default/residual_block4/conv2/filters/Adam<Default/residual_block4/conv2/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Ņ
/Default/residual_block4/conv2/filters/Adam/readIdentity*Default/residual_block4/conv2/filters/Adam*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Ż
>Default/residual_block4/conv2/filters/Adam_1/Initializer/zerosConst*%
valueB>>*    *
dtype0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
ź
,Default/residual_block4/conv2/filters/Adam_1
VariableV2*
shape:>>*
dtype0*
	container *
shared_name *8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
æ
3Default/residual_block4/conv2/filters/Adam_1/AssignAssign,Default/residual_block4/conv2/filters/Adam_1>Default/residual_block4/conv2/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Ö
1Default/residual_block4/conv2/filters/Adam_1/readIdentity,Default/residual_block4/conv2/filters/Adam_1*
T0*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Į
;Default/residual_block4/conv2/biases/Adam/Initializer/zerosConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Ī
)Default/residual_block4/conv2/biases/Adam
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
©
0Default/residual_block4/conv2/biases/Adam/AssignAssign)Default/residual_block4/conv2/biases/Adam;Default/residual_block4/conv2/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Ć
.Default/residual_block4/conv2/biases/Adam/readIdentity)Default/residual_block4/conv2/biases/Adam*
T0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Ć
=Default/residual_block4/conv2/biases/Adam_1/Initializer/zerosConst*
valueB>*    *
dtype0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Š
+Default/residual_block4/conv2/biases/Adam_1
VariableV2*
shape:>*
dtype0*
	container *
shared_name *7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Æ
2Default/residual_block4/conv2/biases/Adam_1/AssignAssign+Default/residual_block4/conv2/biases/Adam_1=Default/residual_block4/conv2/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Ē
0Default/residual_block4/conv2/biases/Adam_1/readIdentity+Default/residual_block4/conv2/biases/Adam_1*
T0*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
Å
1Default/last_layer/filters/Adam/Initializer/zerosConst*%
valueB>K*    *
dtype0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
Ņ
Default/last_layer/filters/Adam
VariableV2*
shape:>K*
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

&Default/last_layer/filters/Adam/AssignAssignDefault/last_layer/filters/Adam1Default/last_layer/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
±
$Default/last_layer/filters/Adam/readIdentityDefault/last_layer/filters/Adam*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
Ē
3Default/last_layer/filters/Adam_1/Initializer/zerosConst*%
valueB>K*    *
dtype0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
Ō
!Default/last_layer/filters/Adam_1
VariableV2*
shape:>K*
dtype0*
	container *
shared_name *-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

(Default/last_layer/filters/Adam_1/AssignAssign!Default/last_layer/filters/Adam_13Default/last_layer/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
µ
&Default/last_layer/filters/Adam_1/readIdentity!Default/last_layer/filters/Adam_1*
T0*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
«
0Default/last_layer/biases/Adam/Initializer/zerosConst*
valueBK*    *
dtype0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ø
Default/last_layer/biases/Adam
VariableV2*
shape:K*
dtype0*
	container *
shared_name *,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ż
%Default/last_layer/biases/Adam/AssignAssignDefault/last_layer/biases/Adam0Default/last_layer/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
¢
#Default/last_layer/biases/Adam/readIdentityDefault/last_layer/biases/Adam*
T0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
­
2Default/last_layer/biases/Adam_1/Initializer/zerosConst*
valueBK*    *
dtype0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ŗ
 Default/last_layer/biases/Adam_1
VariableV2*
shape:K*
dtype0*
	container *
shared_name *,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

'Default/last_layer/biases/Adam_1/AssignAssign Default/last_layer/biases/Adam_12Default/last_layer/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
¦
%Default/last_layer/biases/Adam_1/readIdentity Default/last_layer/biases/Adam_1*
T0*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
×
:Default/down_sampling_layer/filters/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
ä
(Default/down_sampling_layer/filters/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
±
/Default/down_sampling_layer/filters/Adam/AssignAssign(Default/down_sampling_layer/filters/Adam:Default/down_sampling_layer/filters/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Ģ
-Default/down_sampling_layer/filters/Adam/readIdentity(Default/down_sampling_layer/filters/Adam*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Ł
<Default/down_sampling_layer/filters/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
ę
*Default/down_sampling_layer/filters/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
·
1Default/down_sampling_layer/filters/Adam_1/AssignAssign*Default/down_sampling_layer/filters/Adam_1<Default/down_sampling_layer/filters/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Š
/Default/down_sampling_layer/filters/Adam_1/readIdentity*Default/down_sampling_layer/filters/Adam_1*
T0*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
½
9Default/down_sampling_layer/biases/Adam/Initializer/zerosConst*
valueB*    *
dtype0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
Ź
'Default/down_sampling_layer/biases/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
”
.Default/down_sampling_layer/biases/Adam/AssignAssign'Default/down_sampling_layer/biases/Adam9Default/down_sampling_layer/biases/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
½
,Default/down_sampling_layer/biases/Adam/readIdentity'Default/down_sampling_layer/biases/Adam*
T0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
æ
;Default/down_sampling_layer/biases/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
Ģ
)Default/down_sampling_layer/biases/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
§
0Default/down_sampling_layer/biases/Adam_1/AssignAssign)Default/down_sampling_layer/biases/Adam_1;Default/down_sampling_layer/biases/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
Į
.Default/down_sampling_layer/biases/Adam_1/readIdentity)Default/down_sampling_layer/biases/Adam_1*
T0*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *·Ń8*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 
¼
1Adam/update_Default/first_layer/filters/ApplyAdam	ApplyAdamDefault/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_13_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 
Ø
0Adam/update_Default/first_layer/biases/ApplyAdam	ApplyAdamDefault/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_18_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 
ī
;Adam/update_Default/residual_block0/conv1/filters/ApplyAdam	ApplyAdam%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_14_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "
Ś
:Adam/update_Default/residual_block0/conv1/biases/ApplyAdam	ApplyAdam$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_19_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"
ī
;Adam/update_Default/residual_block0/conv2/filters/ApplyAdam	ApplyAdam%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_15_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""
Ś
:Adam/update_Default/residual_block0/conv2/biases/ApplyAdam	ApplyAdam$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_20_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"
ī
;Adam/update_Default/residual_block1/conv1/filters/ApplyAdam	ApplyAdam%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_16_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&
Ś
:Adam/update_Default/residual_block1/conv1/biases/ApplyAdam	ApplyAdam$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_22_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&
ī
;Adam/update_Default/residual_block1/conv2/filters/ApplyAdam	ApplyAdam%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_17_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&
Ś
:Adam/update_Default/residual_block1/conv2/biases/ApplyAdam	ApplyAdam$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_23_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&
ī
;Adam/update_Default/residual_block2/conv1/filters/ApplyAdam	ApplyAdam%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_18_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,
Ś
:Adam/update_Default/residual_block2/conv1/biases/ApplyAdam	ApplyAdam$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_25_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,
ī
;Adam/update_Default/residual_block2/conv2/filters/ApplyAdam	ApplyAdam%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_19_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,
Ś
:Adam/update_Default/residual_block2/conv2/biases/ApplyAdam	ApplyAdam$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_26_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,
ī
;Adam/update_Default/residual_block3/conv1/filters/ApplyAdam	ApplyAdam%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_20_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4
Ś
:Adam/update_Default/residual_block3/conv1/biases/ApplyAdam	ApplyAdam$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_28_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4
ī
;Adam/update_Default/residual_block3/conv2/filters/ApplyAdam	ApplyAdam%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_21_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44
Ś
:Adam/update_Default/residual_block3/conv2/biases/ApplyAdam	ApplyAdam$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_29_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4
ī
;Adam/update_Default/residual_block4/conv1/filters/ApplyAdam	ApplyAdam%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_22_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>
Ś
:Adam/update_Default/residual_block4/conv1/biases/ApplyAdam	ApplyAdam$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_31_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>
ī
;Adam/update_Default/residual_block4/conv2/filters/ApplyAdam	ApplyAdam%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_23_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
Ś
:Adam/update_Default/residual_block4/conv2/biases/ApplyAdam	ApplyAdam$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_32_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>
·
0Adam/update_Default/last_layer/filters/ApplyAdam	ApplyAdamDefault/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_24_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K
£
/Adam/update_Default/last_layer/biases/ApplyAdam	ApplyAdamDefault/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_34_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K
ä
9Adam/update_Default/down_sampling_layer/filters/ApplyAdam	ApplyAdam#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/Conv2D_25_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:
Š
8Adam/update_Default/down_sampling_layer/biases/ApplyAdam	ApplyAdam"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/add_35_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

Adam/mulMulbeta1_power/read
Adam/beta12^Adam/update_Default/first_layer/filters/ApplyAdam1^Adam/update_Default/first_layer/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv2/biases/ApplyAdam1^Adam/update_Default/last_layer/filters/ApplyAdam0^Adam/update_Default/last_layer/biases/ApplyAdam:^Adam/update_Default/down_sampling_layer/filters/ApplyAdam9^Adam/update_Default/down_sampling_layer/biases/ApplyAdam*
T0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
¦
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta22^Adam/update_Default/first_layer/filters/ApplyAdam1^Adam/update_Default/first_layer/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv2/biases/ApplyAdam1^Adam/update_Default/last_layer/filters/ApplyAdam0^Adam/update_Default/last_layer/biases/ApplyAdam:^Adam/update_Default/down_sampling_layer/filters/ApplyAdam9^Adam/update_Default/down_sampling_layer/biases/ApplyAdam*
T0*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
Ŗ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
»
AdamNoOp2^Adam/update_Default/first_layer/filters/ApplyAdam1^Adam/update_Default/first_layer/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block0/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block0/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block1/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block1/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block2/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block2/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block3/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block3/conv2/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv1/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv1/biases/ApplyAdam<^Adam/update_Default/residual_block4/conv2/filters/ApplyAdam;^Adam/update_Default/residual_block4/conv2/biases/ApplyAdam1^Adam/update_Default/last_layer/filters/ApplyAdam0^Adam/update_Default/last_layer/biases/ApplyAdam:^Adam/update_Default/down_sampling_layer/filters/ApplyAdam9^Adam/update_Default/down_sampling_layer/biases/ApplyAdam^Adam/Assign^Adam/Assign_1
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
å
save/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
¼
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ī
save/AssignAssign"Default/down_sampling_layer/biasessave/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_3Assign#Default/down_sampling_layer/filterssave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
å
save/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ē
save/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ā
save/Assign_6AssignDefault/first_layer/biasessave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ē
save/Assign_7AssignDefault/first_layer/biases/Adamsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
É
save/Assign_8Assign!Default/first_layer/biases/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_9AssignDefault/first_layer/filterssave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

save/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_10Assign Default/first_layer/filters/Adamsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

save/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save/Assign_11Assign"Default/first_layer/filters/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

save/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ā
save/Assign_12AssignDefault/last_layer/biasessave/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

save/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ē
save/Assign_13AssignDefault/last_layer/biases/Adamsave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

save/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
É
save/Assign_14Assign Default/last_layer/biases/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

save/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_15AssignDefault/last_layer/filterssave/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

save/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Õ
save/Assign_16AssignDefault/last_layer/filters/Adamsave/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

save/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
×
save/Assign_17Assign!Default/last_layer/filters/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

save/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_18Assign$Default/residual_block0/conv1/biasessave/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

save/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

save/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

save/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_21Assign%Default/residual_block0/conv1/filterssave/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

save/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

save/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

save/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_24Assign$Default/residual_block0/conv2/biasessave/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

save/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

save/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

save/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_27Assign%Default/residual_block0/conv2/filterssave/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

save/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

save/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

save/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_30Assign$Default/residual_block1/conv1/biasessave/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

save/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

save/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

save/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_33Assign%Default/residual_block1/conv1/filterssave/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

save/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

save/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

save/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_36Assign$Default/residual_block1/conv2/biasessave/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

save/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

save/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

save/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_39Assign%Default/residual_block1/conv2/filterssave/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

save/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

save/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

save/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_42Assign$Default/residual_block2/conv1/biasessave/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

save/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

save/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_44	RestoreV2
save/Constsave/RestoreV2_44/tensor_names"save/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

save/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_45	RestoreV2
save/Constsave/RestoreV2_45/tensor_names"save/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_45Assign%Default/residual_block2/conv1/filterssave/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

save/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_46	RestoreV2
save/Constsave/RestoreV2_46/tensor_names"save/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

save/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_47	RestoreV2
save/Constsave/RestoreV2_47/tensor_names"save/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

save/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_48	RestoreV2
save/Constsave/RestoreV2_48/tensor_names"save/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_48Assign$Default/residual_block2/conv2/biasessave/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

save/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_49	RestoreV2
save/Constsave/RestoreV2_49/tensor_names"save/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

save/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_50	RestoreV2
save/Constsave/RestoreV2_50/tensor_names"save/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

save/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_51	RestoreV2
save/Constsave/RestoreV2_51/tensor_names"save/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_51Assign%Default/residual_block2/conv2/filterssave/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

save/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_52	RestoreV2
save/Constsave/RestoreV2_52/tensor_names"save/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

save/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_53	RestoreV2
save/Constsave/RestoreV2_53/tensor_names"save/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

save/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_54	RestoreV2
save/Constsave/RestoreV2_54/tensor_names"save/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_54Assign$Default/residual_block3/conv1/biasessave/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

save/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_55	RestoreV2
save/Constsave/RestoreV2_55/tensor_names"save/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

save/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_56	RestoreV2
save/Constsave/RestoreV2_56/tensor_names"save/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

save/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_57	RestoreV2
save/Constsave/RestoreV2_57/tensor_names"save/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_57Assign%Default/residual_block3/conv1/filterssave/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

save/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_58	RestoreV2
save/Constsave/RestoreV2_58/tensor_names"save/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

save/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_59	RestoreV2
save/Constsave/RestoreV2_59/tensor_names"save/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

save/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_60	RestoreV2
save/Constsave/RestoreV2_60/tensor_names"save/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_60Assign$Default/residual_block3/conv2/biasessave/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

save/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_61	RestoreV2
save/Constsave/RestoreV2_61/tensor_names"save/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

save/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_62	RestoreV2
save/Constsave/RestoreV2_62/tensor_names"save/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

save/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_63	RestoreV2
save/Constsave/RestoreV2_63/tensor_names"save/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_63Assign%Default/residual_block3/conv2/filterssave/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

save/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_64	RestoreV2
save/Constsave/RestoreV2_64/tensor_names"save/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

save/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_65	RestoreV2
save/Constsave/RestoreV2_65/tensor_names"save/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

save/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_66	RestoreV2
save/Constsave/RestoreV2_66/tensor_names"save/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_66Assign$Default/residual_block4/conv1/biasessave/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

save/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_67	RestoreV2
save/Constsave/RestoreV2_67/tensor_names"save/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

save/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_68	RestoreV2
save/Constsave/RestoreV2_68/tensor_names"save/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

save/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_69	RestoreV2
save/Constsave/RestoreV2_69/tensor_names"save/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_69Assign%Default/residual_block4/conv1/filterssave/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

save/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_70	RestoreV2
save/Constsave/RestoreV2_70/tensor_names"save/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

save/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_71	RestoreV2
save/Constsave/RestoreV2_71/tensor_names"save/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

save/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_72	RestoreV2
save/Constsave/RestoreV2_72/tensor_names"save/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ų
save/Assign_72Assign$Default/residual_block4/conv2/biasessave/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

save/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_73	RestoreV2
save/Constsave/RestoreV2_73/tensor_names"save/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

save/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_74	RestoreV2
save/Constsave/RestoreV2_74/tensor_names"save/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ß
save/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

save/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
k
"save/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_75	RestoreV2
save/Constsave/RestoreV2_75/tensor_names"save/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ę
save/Assign_75Assign%Default/residual_block4/conv2/filterssave/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

save/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
k
"save/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_76	RestoreV2
save/Constsave/RestoreV2_76/tensor_names"save/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

save/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
k
"save/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_77	RestoreV2
save/Constsave/RestoreV2_77/tensor_names"save/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ķ
save/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
r
save/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_78	RestoreV2
save/Constsave/RestoreV2_78/tensor_names"save/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
²
save/Assign_78Assignbeta1_powersave/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
r
save/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
k
"save/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save/RestoreV2_79	RestoreV2
save/Constsave/RestoreV2_79/tensor_names"save/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
²
save/Assign_79Assignbeta2_powersave/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
Ü

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79

initNoOp#^Default/first_layer/filters/Assign"^Default/first_layer/biases/Assign-^Default/residual_block0/conv1/filters/Assign,^Default/residual_block0/conv1/biases/Assign-^Default/residual_block0/conv2/filters/Assign,^Default/residual_block0/conv2/biases/Assign-^Default/residual_block1/conv1/filters/Assign,^Default/residual_block1/conv1/biases/Assign-^Default/residual_block1/conv2/filters/Assign,^Default/residual_block1/conv2/biases/Assign-^Default/residual_block2/conv1/filters/Assign,^Default/residual_block2/conv1/biases/Assign-^Default/residual_block2/conv2/filters/Assign,^Default/residual_block2/conv2/biases/Assign-^Default/residual_block3/conv1/filters/Assign,^Default/residual_block3/conv1/biases/Assign-^Default/residual_block3/conv2/filters/Assign,^Default/residual_block3/conv2/biases/Assign-^Default/residual_block4/conv1/filters/Assign,^Default/residual_block4/conv1/biases/Assign-^Default/residual_block4/conv2/filters/Assign,^Default/residual_block4/conv2/biases/Assign"^Default/last_layer/filters/Assign!^Default/last_layer/biases/Assign+^Default/down_sampling_layer/filters/Assign*^Default/down_sampling_layer/biases/Assign^beta1_power/Assign^beta2_power/Assign(^Default/first_layer/filters/Adam/Assign*^Default/first_layer/filters/Adam_1/Assign'^Default/first_layer/biases/Adam/Assign)^Default/first_layer/biases/Adam_1/Assign2^Default/residual_block0/conv1/filters/Adam/Assign4^Default/residual_block0/conv1/filters/Adam_1/Assign1^Default/residual_block0/conv1/biases/Adam/Assign3^Default/residual_block0/conv1/biases/Adam_1/Assign2^Default/residual_block0/conv2/filters/Adam/Assign4^Default/residual_block0/conv2/filters/Adam_1/Assign1^Default/residual_block0/conv2/biases/Adam/Assign3^Default/residual_block0/conv2/biases/Adam_1/Assign2^Default/residual_block1/conv1/filters/Adam/Assign4^Default/residual_block1/conv1/filters/Adam_1/Assign1^Default/residual_block1/conv1/biases/Adam/Assign3^Default/residual_block1/conv1/biases/Adam_1/Assign2^Default/residual_block1/conv2/filters/Adam/Assign4^Default/residual_block1/conv2/filters/Adam_1/Assign1^Default/residual_block1/conv2/biases/Adam/Assign3^Default/residual_block1/conv2/biases/Adam_1/Assign2^Default/residual_block2/conv1/filters/Adam/Assign4^Default/residual_block2/conv1/filters/Adam_1/Assign1^Default/residual_block2/conv1/biases/Adam/Assign3^Default/residual_block2/conv1/biases/Adam_1/Assign2^Default/residual_block2/conv2/filters/Adam/Assign4^Default/residual_block2/conv2/filters/Adam_1/Assign1^Default/residual_block2/conv2/biases/Adam/Assign3^Default/residual_block2/conv2/biases/Adam_1/Assign2^Default/residual_block3/conv1/filters/Adam/Assign4^Default/residual_block3/conv1/filters/Adam_1/Assign1^Default/residual_block3/conv1/biases/Adam/Assign3^Default/residual_block3/conv1/biases/Adam_1/Assign2^Default/residual_block3/conv2/filters/Adam/Assign4^Default/residual_block3/conv2/filters/Adam_1/Assign1^Default/residual_block3/conv2/biases/Adam/Assign3^Default/residual_block3/conv2/biases/Adam_1/Assign2^Default/residual_block4/conv1/filters/Adam/Assign4^Default/residual_block4/conv1/filters/Adam_1/Assign1^Default/residual_block4/conv1/biases/Adam/Assign3^Default/residual_block4/conv1/biases/Adam_1/Assign2^Default/residual_block4/conv2/filters/Adam/Assign4^Default/residual_block4/conv2/filters/Adam_1/Assign1^Default/residual_block4/conv2/biases/Adam/Assign3^Default/residual_block4/conv2/biases/Adam_1/Assign'^Default/last_layer/filters/Adam/Assign)^Default/last_layer/filters/Adam_1/Assign&^Default/last_layer/biases/Adam/Assign(^Default/last_layer/biases/Adam_1/Assign0^Default/down_sampling_layer/filters/Adam/Assign2^Default/down_sampling_layer/filters/Adam_1/Assign/^Default/down_sampling_layer/biases/Adam/Assign1^Default/down_sampling_layer/biases/Adam_1/Assign
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_51b79d45cd1a4f4dbf723cd6841695e9/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ē
save_1/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_1/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_1/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_1/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_1/AssignAssign"Default/down_sampling_layer/biasessave_1/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_1/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_1/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_1/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_1/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_1/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_1/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_1/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_1/Assign_3Assign#Default/down_sampling_layer/filterssave_1/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_1/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_1/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_1/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_1/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_1/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_1/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_1/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_1/Assign_6AssignDefault/first_layer/biasessave_1/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_1/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_1/Assign_7AssignDefault/first_layer/biases/Adamsave_1/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_1/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_1/Assign_8Assign!Default/first_layer/biases/Adam_1save_1/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_1/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_1/Assign_9AssignDefault/first_layer/filterssave_1/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_1/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_1/Assign_10Assign Default/first_layer/filters/Adamsave_1/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_1/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_1/Assign_11Assign"Default/first_layer/filters/Adam_1save_1/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_1/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_1/Assign_12AssignDefault/last_layer/biasessave_1/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_1/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_1/Assign_13AssignDefault/last_layer/biases/Adamsave_1/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_1/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_1/Assign_14Assign Default/last_layer/biases/Adam_1save_1/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_1/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_1/Assign_15AssignDefault/last_layer/filterssave_1/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_1/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_1/Assign_16AssignDefault/last_layer/filters/Adamsave_1/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_1/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_1/Assign_17Assign!Default/last_layer/filters/Adam_1save_1/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_1/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_18Assign$Default/residual_block0/conv1/biasessave_1/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_1/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_1/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_1/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_1/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_1/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_21	RestoreV2save_1/Const save_1/RestoreV2_21/tensor_names$save_1/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_21Assign%Default/residual_block0/conv1/filterssave_1/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_1/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_22	RestoreV2save_1/Const save_1/RestoreV2_22/tensor_names$save_1/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_1/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_1/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_23	RestoreV2save_1/Const save_1/RestoreV2_23/tensor_names$save_1/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_1/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_1/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_24	RestoreV2save_1/Const save_1/RestoreV2_24/tensor_names$save_1/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_24Assign$Default/residual_block0/conv2/biasessave_1/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_1/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_25	RestoreV2save_1/Const save_1/RestoreV2_25/tensor_names$save_1/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_1/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_1/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_26	RestoreV2save_1/Const save_1/RestoreV2_26/tensor_names$save_1/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_1/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_1/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_27	RestoreV2save_1/Const save_1/RestoreV2_27/tensor_names$save_1/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_27Assign%Default/residual_block0/conv2/filterssave_1/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_1/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_28	RestoreV2save_1/Const save_1/RestoreV2_28/tensor_names$save_1/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_1/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_1/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_29	RestoreV2save_1/Const save_1/RestoreV2_29/tensor_names$save_1/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_1/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_1/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_30	RestoreV2save_1/Const save_1/RestoreV2_30/tensor_names$save_1/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_30Assign$Default/residual_block1/conv1/biasessave_1/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_1/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_31	RestoreV2save_1/Const save_1/RestoreV2_31/tensor_names$save_1/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_1/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_1/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_32	RestoreV2save_1/Const save_1/RestoreV2_32/tensor_names$save_1/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_1/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_1/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_33	RestoreV2save_1/Const save_1/RestoreV2_33/tensor_names$save_1/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_33Assign%Default/residual_block1/conv1/filterssave_1/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_1/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_34	RestoreV2save_1/Const save_1/RestoreV2_34/tensor_names$save_1/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_1/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_1/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_35	RestoreV2save_1/Const save_1/RestoreV2_35/tensor_names$save_1/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_1/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_1/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_36	RestoreV2save_1/Const save_1/RestoreV2_36/tensor_names$save_1/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_36Assign$Default/residual_block1/conv2/biasessave_1/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_1/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_37	RestoreV2save_1/Const save_1/RestoreV2_37/tensor_names$save_1/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_1/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_1/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_38	RestoreV2save_1/Const save_1/RestoreV2_38/tensor_names$save_1/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_1/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_1/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_39	RestoreV2save_1/Const save_1/RestoreV2_39/tensor_names$save_1/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_39Assign%Default/residual_block1/conv2/filterssave_1/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_1/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_40	RestoreV2save_1/Const save_1/RestoreV2_40/tensor_names$save_1/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_1/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_1/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_41	RestoreV2save_1/Const save_1/RestoreV2_41/tensor_names$save_1/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_1/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_1/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_42	RestoreV2save_1/Const save_1/RestoreV2_42/tensor_names$save_1/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_42Assign$Default/residual_block2/conv1/biasessave_1/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_1/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_43	RestoreV2save_1/Const save_1/RestoreV2_43/tensor_names$save_1/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_1/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_1/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_44	RestoreV2save_1/Const save_1/RestoreV2_44/tensor_names$save_1/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_1/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_1/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_45	RestoreV2save_1/Const save_1/RestoreV2_45/tensor_names$save_1/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_45Assign%Default/residual_block2/conv1/filterssave_1/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_1/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_46	RestoreV2save_1/Const save_1/RestoreV2_46/tensor_names$save_1/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_1/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_1/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_47	RestoreV2save_1/Const save_1/RestoreV2_47/tensor_names$save_1/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_1/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_1/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_48	RestoreV2save_1/Const save_1/RestoreV2_48/tensor_names$save_1/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_48Assign$Default/residual_block2/conv2/biasessave_1/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_1/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_49	RestoreV2save_1/Const save_1/RestoreV2_49/tensor_names$save_1/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_1/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_1/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_50	RestoreV2save_1/Const save_1/RestoreV2_50/tensor_names$save_1/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_1/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_1/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_51	RestoreV2save_1/Const save_1/RestoreV2_51/tensor_names$save_1/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_51Assign%Default/residual_block2/conv2/filterssave_1/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_1/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_52	RestoreV2save_1/Const save_1/RestoreV2_52/tensor_names$save_1/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_1/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_1/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_53	RestoreV2save_1/Const save_1/RestoreV2_53/tensor_names$save_1/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_1/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_1/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_54	RestoreV2save_1/Const save_1/RestoreV2_54/tensor_names$save_1/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_54Assign$Default/residual_block3/conv1/biasessave_1/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_1/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_55	RestoreV2save_1/Const save_1/RestoreV2_55/tensor_names$save_1/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_1/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_1/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_56	RestoreV2save_1/Const save_1/RestoreV2_56/tensor_names$save_1/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_1/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_1/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_57	RestoreV2save_1/Const save_1/RestoreV2_57/tensor_names$save_1/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_57Assign%Default/residual_block3/conv1/filterssave_1/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_1/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_58	RestoreV2save_1/Const save_1/RestoreV2_58/tensor_names$save_1/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_1/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_1/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_59	RestoreV2save_1/Const save_1/RestoreV2_59/tensor_names$save_1/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_1/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_1/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_60	RestoreV2save_1/Const save_1/RestoreV2_60/tensor_names$save_1/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_60Assign$Default/residual_block3/conv2/biasessave_1/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_1/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_61	RestoreV2save_1/Const save_1/RestoreV2_61/tensor_names$save_1/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_1/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_1/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_62	RestoreV2save_1/Const save_1/RestoreV2_62/tensor_names$save_1/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_1/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_1/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_63	RestoreV2save_1/Const save_1/RestoreV2_63/tensor_names$save_1/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_63Assign%Default/residual_block3/conv2/filterssave_1/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_1/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_64	RestoreV2save_1/Const save_1/RestoreV2_64/tensor_names$save_1/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_1/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_1/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_65	RestoreV2save_1/Const save_1/RestoreV2_65/tensor_names$save_1/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_1/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_1/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_66	RestoreV2save_1/Const save_1/RestoreV2_66/tensor_names$save_1/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_66Assign$Default/residual_block4/conv1/biasessave_1/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_1/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_67	RestoreV2save_1/Const save_1/RestoreV2_67/tensor_names$save_1/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_1/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_1/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_68	RestoreV2save_1/Const save_1/RestoreV2_68/tensor_names$save_1/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_1/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_1/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_69	RestoreV2save_1/Const save_1/RestoreV2_69/tensor_names$save_1/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_69Assign%Default/residual_block4/conv1/filterssave_1/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_1/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_70	RestoreV2save_1/Const save_1/RestoreV2_70/tensor_names$save_1/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_1/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_1/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_71	RestoreV2save_1/Const save_1/RestoreV2_71/tensor_names$save_1/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_1/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_1/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_72	RestoreV2save_1/Const save_1/RestoreV2_72/tensor_names$save_1/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_1/Assign_72Assign$Default/residual_block4/conv2/biasessave_1/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_1/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_73	RestoreV2save_1/Const save_1/RestoreV2_73/tensor_names$save_1/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_1/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_1/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_1/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_74	RestoreV2save_1/Const save_1/RestoreV2_74/tensor_names$save_1/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_1/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_1/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_1/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_75	RestoreV2save_1/Const save_1/RestoreV2_75/tensor_names$save_1/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_75Assign%Default/residual_block4/conv2/filterssave_1/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_1/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_76	RestoreV2save_1/Const save_1/RestoreV2_76/tensor_names$save_1/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_1/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_1/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_1/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_77	RestoreV2save_1/Const save_1/RestoreV2_77/tensor_names$save_1/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_1/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_1/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_1/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_78	RestoreV2save_1/Const save_1/RestoreV2_78/tensor_names$save_1/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_1/Assign_78Assignbeta1_powersave_1/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_1/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_1/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_1/RestoreV2_79	RestoreV2save_1/Const save_1/RestoreV2_79/tensor_names$save_1/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_1/Assign_79Assignbeta2_powersave_1/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79
1
save_1/restore_allNoOp^save_1/restore_shard
R
save_2/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_cf53c1bde7424294a9088f803e40934d/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
ē
save_2/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_2/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/control_dependency^save_2/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_2/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_2/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_2/AssignAssign"Default/down_sampling_layer/biasessave_2/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_2/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_1	RestoreV2save_2/Constsave_2/RestoreV2_1/tensor_names#save_2/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_2/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_2/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_2/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_2	RestoreV2save_2/Constsave_2/RestoreV2_2/tensor_names#save_2/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_2/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_2/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_2/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_3	RestoreV2save_2/Constsave_2/RestoreV2_3/tensor_names#save_2/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_2/Assign_3Assign#Default/down_sampling_layer/filterssave_2/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_2/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_4	RestoreV2save_2/Constsave_2/RestoreV2_4/tensor_names#save_2/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_2/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_2/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_2/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_5	RestoreV2save_2/Constsave_2/RestoreV2_5/tensor_names#save_2/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_2/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_2/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_2/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_6	RestoreV2save_2/Constsave_2/RestoreV2_6/tensor_names#save_2/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_2/Assign_6AssignDefault/first_layer/biasessave_2/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_2/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_7	RestoreV2save_2/Constsave_2/RestoreV2_7/tensor_names#save_2/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_2/Assign_7AssignDefault/first_layer/biases/Adamsave_2/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_2/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_8	RestoreV2save_2/Constsave_2/RestoreV2_8/tensor_names#save_2/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_2/Assign_8Assign!Default/first_layer/biases/Adam_1save_2/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_2/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_2/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_2/RestoreV2_9	RestoreV2save_2/Constsave_2/RestoreV2_9/tensor_names#save_2/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_2/Assign_9AssignDefault/first_layer/filterssave_2/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_2/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_10	RestoreV2save_2/Const save_2/RestoreV2_10/tensor_names$save_2/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_2/Assign_10Assign Default/first_layer/filters/Adamsave_2/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_2/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_11	RestoreV2save_2/Const save_2/RestoreV2_11/tensor_names$save_2/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_2/Assign_11Assign"Default/first_layer/filters/Adam_1save_2/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_2/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_12	RestoreV2save_2/Const save_2/RestoreV2_12/tensor_names$save_2/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_2/Assign_12AssignDefault/last_layer/biasessave_2/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_2/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_13	RestoreV2save_2/Const save_2/RestoreV2_13/tensor_names$save_2/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_2/Assign_13AssignDefault/last_layer/biases/Adamsave_2/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_2/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_14	RestoreV2save_2/Const save_2/RestoreV2_14/tensor_names$save_2/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_2/Assign_14Assign Default/last_layer/biases/Adam_1save_2/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_2/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_15	RestoreV2save_2/Const save_2/RestoreV2_15/tensor_names$save_2/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_2/Assign_15AssignDefault/last_layer/filterssave_2/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_2/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_16	RestoreV2save_2/Const save_2/RestoreV2_16/tensor_names$save_2/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_2/Assign_16AssignDefault/last_layer/filters/Adamsave_2/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_2/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_17	RestoreV2save_2/Const save_2/RestoreV2_17/tensor_names$save_2/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_2/Assign_17Assign!Default/last_layer/filters/Adam_1save_2/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_2/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_18	RestoreV2save_2/Const save_2/RestoreV2_18/tensor_names$save_2/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_18Assign$Default/residual_block0/conv1/biasessave_2/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_2/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_19	RestoreV2save_2/Const save_2/RestoreV2_19/tensor_names$save_2/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_2/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_2/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_20	RestoreV2save_2/Const save_2/RestoreV2_20/tensor_names$save_2/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_2/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_2/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_21	RestoreV2save_2/Const save_2/RestoreV2_21/tensor_names$save_2/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_21Assign%Default/residual_block0/conv1/filterssave_2/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_2/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_22	RestoreV2save_2/Const save_2/RestoreV2_22/tensor_names$save_2/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_2/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_2/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_23	RestoreV2save_2/Const save_2/RestoreV2_23/tensor_names$save_2/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_2/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_2/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_24	RestoreV2save_2/Const save_2/RestoreV2_24/tensor_names$save_2/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_24Assign$Default/residual_block0/conv2/biasessave_2/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_2/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_25	RestoreV2save_2/Const save_2/RestoreV2_25/tensor_names$save_2/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_2/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_2/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_26	RestoreV2save_2/Const save_2/RestoreV2_26/tensor_names$save_2/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_2/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_2/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_27	RestoreV2save_2/Const save_2/RestoreV2_27/tensor_names$save_2/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_27Assign%Default/residual_block0/conv2/filterssave_2/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_2/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_28	RestoreV2save_2/Const save_2/RestoreV2_28/tensor_names$save_2/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_2/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_2/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_29	RestoreV2save_2/Const save_2/RestoreV2_29/tensor_names$save_2/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_2/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_2/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_30	RestoreV2save_2/Const save_2/RestoreV2_30/tensor_names$save_2/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_30Assign$Default/residual_block1/conv1/biasessave_2/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_2/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_31	RestoreV2save_2/Const save_2/RestoreV2_31/tensor_names$save_2/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_2/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_2/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_32	RestoreV2save_2/Const save_2/RestoreV2_32/tensor_names$save_2/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_2/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_2/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_33	RestoreV2save_2/Const save_2/RestoreV2_33/tensor_names$save_2/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_33Assign%Default/residual_block1/conv1/filterssave_2/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_2/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_34	RestoreV2save_2/Const save_2/RestoreV2_34/tensor_names$save_2/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_2/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_2/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_35	RestoreV2save_2/Const save_2/RestoreV2_35/tensor_names$save_2/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_2/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_2/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_36	RestoreV2save_2/Const save_2/RestoreV2_36/tensor_names$save_2/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_36Assign$Default/residual_block1/conv2/biasessave_2/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_2/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_37	RestoreV2save_2/Const save_2/RestoreV2_37/tensor_names$save_2/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_2/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_2/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_38	RestoreV2save_2/Const save_2/RestoreV2_38/tensor_names$save_2/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_2/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_2/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_39	RestoreV2save_2/Const save_2/RestoreV2_39/tensor_names$save_2/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_39Assign%Default/residual_block1/conv2/filterssave_2/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_2/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_40	RestoreV2save_2/Const save_2/RestoreV2_40/tensor_names$save_2/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_2/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_2/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_41	RestoreV2save_2/Const save_2/RestoreV2_41/tensor_names$save_2/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_2/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_2/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_42	RestoreV2save_2/Const save_2/RestoreV2_42/tensor_names$save_2/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_42Assign$Default/residual_block2/conv1/biasessave_2/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_2/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_43	RestoreV2save_2/Const save_2/RestoreV2_43/tensor_names$save_2/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_2/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_2/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_44	RestoreV2save_2/Const save_2/RestoreV2_44/tensor_names$save_2/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_2/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_2/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_45	RestoreV2save_2/Const save_2/RestoreV2_45/tensor_names$save_2/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_45Assign%Default/residual_block2/conv1/filterssave_2/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_2/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_46	RestoreV2save_2/Const save_2/RestoreV2_46/tensor_names$save_2/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_2/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_2/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_47	RestoreV2save_2/Const save_2/RestoreV2_47/tensor_names$save_2/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_2/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_2/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_48	RestoreV2save_2/Const save_2/RestoreV2_48/tensor_names$save_2/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_48Assign$Default/residual_block2/conv2/biasessave_2/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_2/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_49	RestoreV2save_2/Const save_2/RestoreV2_49/tensor_names$save_2/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_2/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_2/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_50	RestoreV2save_2/Const save_2/RestoreV2_50/tensor_names$save_2/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_2/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_2/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_51	RestoreV2save_2/Const save_2/RestoreV2_51/tensor_names$save_2/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_51Assign%Default/residual_block2/conv2/filterssave_2/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_2/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_52	RestoreV2save_2/Const save_2/RestoreV2_52/tensor_names$save_2/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_2/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_2/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_53	RestoreV2save_2/Const save_2/RestoreV2_53/tensor_names$save_2/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_2/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_2/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_54	RestoreV2save_2/Const save_2/RestoreV2_54/tensor_names$save_2/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_54Assign$Default/residual_block3/conv1/biasessave_2/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_2/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_55	RestoreV2save_2/Const save_2/RestoreV2_55/tensor_names$save_2/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_2/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_2/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_56	RestoreV2save_2/Const save_2/RestoreV2_56/tensor_names$save_2/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_2/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_2/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_57	RestoreV2save_2/Const save_2/RestoreV2_57/tensor_names$save_2/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_57Assign%Default/residual_block3/conv1/filterssave_2/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_2/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_58	RestoreV2save_2/Const save_2/RestoreV2_58/tensor_names$save_2/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_2/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_2/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_59	RestoreV2save_2/Const save_2/RestoreV2_59/tensor_names$save_2/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_2/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_2/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_60	RestoreV2save_2/Const save_2/RestoreV2_60/tensor_names$save_2/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_60Assign$Default/residual_block3/conv2/biasessave_2/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_2/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_61	RestoreV2save_2/Const save_2/RestoreV2_61/tensor_names$save_2/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_2/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_2/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_62	RestoreV2save_2/Const save_2/RestoreV2_62/tensor_names$save_2/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_2/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_2/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_63	RestoreV2save_2/Const save_2/RestoreV2_63/tensor_names$save_2/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_63Assign%Default/residual_block3/conv2/filterssave_2/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_2/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_64	RestoreV2save_2/Const save_2/RestoreV2_64/tensor_names$save_2/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_2/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_2/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_65	RestoreV2save_2/Const save_2/RestoreV2_65/tensor_names$save_2/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_2/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_2/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_66	RestoreV2save_2/Const save_2/RestoreV2_66/tensor_names$save_2/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_66Assign$Default/residual_block4/conv1/biasessave_2/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_2/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_67	RestoreV2save_2/Const save_2/RestoreV2_67/tensor_names$save_2/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_2/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_2/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_68	RestoreV2save_2/Const save_2/RestoreV2_68/tensor_names$save_2/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_2/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_2/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_69	RestoreV2save_2/Const save_2/RestoreV2_69/tensor_names$save_2/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_69Assign%Default/residual_block4/conv1/filterssave_2/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_2/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_70	RestoreV2save_2/Const save_2/RestoreV2_70/tensor_names$save_2/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_2/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_2/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_71	RestoreV2save_2/Const save_2/RestoreV2_71/tensor_names$save_2/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_2/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_2/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_72	RestoreV2save_2/Const save_2/RestoreV2_72/tensor_names$save_2/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_2/Assign_72Assign$Default/residual_block4/conv2/biasessave_2/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_2/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_73	RestoreV2save_2/Const save_2/RestoreV2_73/tensor_names$save_2/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_2/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_2/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_2/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_74	RestoreV2save_2/Const save_2/RestoreV2_74/tensor_names$save_2/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_2/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_2/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_2/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_75	RestoreV2save_2/Const save_2/RestoreV2_75/tensor_names$save_2/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_2/Assign_75Assign%Default/residual_block4/conv2/filterssave_2/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_2/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_76	RestoreV2save_2/Const save_2/RestoreV2_76/tensor_names$save_2/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_2/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_2/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_2/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_77	RestoreV2save_2/Const save_2/RestoreV2_77/tensor_names$save_2/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_2/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_2/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_2/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_78	RestoreV2save_2/Const save_2/RestoreV2_78/tensor_names$save_2/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_2/Assign_78Assignbeta1_powersave_2/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_2/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_2/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_2/RestoreV2_79	RestoreV2save_2/Const save_2/RestoreV2_79/tensor_names$save_2/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_2/Assign_79Assignbeta2_powersave_2/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_68^save_2/Assign_69^save_2/Assign_70^save_2/Assign_71^save_2/Assign_72^save_2/Assign_73^save_2/Assign_74^save_2/Assign_75^save_2/Assign_76^save_2/Assign_77^save_2/Assign_78^save_2/Assign_79
1
save_2/restore_allNoOp^save_2/restore_shard
R
save_3/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_1cd0979490b34770a0e901e09fe45573/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
ē
save_3/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_3/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/control_dependency^save_3/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_3/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_3/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_3/AssignAssign"Default/down_sampling_layer/biasessave_3/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_3/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_1	RestoreV2save_3/Constsave_3/RestoreV2_1/tensor_names#save_3/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_3/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_3/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_3/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_2	RestoreV2save_3/Constsave_3/RestoreV2_2/tensor_names#save_3/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_3/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_3/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_3/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_3	RestoreV2save_3/Constsave_3/RestoreV2_3/tensor_names#save_3/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_3/Assign_3Assign#Default/down_sampling_layer/filterssave_3/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_3/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_4	RestoreV2save_3/Constsave_3/RestoreV2_4/tensor_names#save_3/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_3/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_3/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_3/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_5	RestoreV2save_3/Constsave_3/RestoreV2_5/tensor_names#save_3/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_3/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_3/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_3/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_6	RestoreV2save_3/Constsave_3/RestoreV2_6/tensor_names#save_3/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_3/Assign_6AssignDefault/first_layer/biasessave_3/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_3/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_7	RestoreV2save_3/Constsave_3/RestoreV2_7/tensor_names#save_3/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_3/Assign_7AssignDefault/first_layer/biases/Adamsave_3/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_3/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_8	RestoreV2save_3/Constsave_3/RestoreV2_8/tensor_names#save_3/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_3/Assign_8Assign!Default/first_layer/biases/Adam_1save_3/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_3/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_3/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_3/RestoreV2_9	RestoreV2save_3/Constsave_3/RestoreV2_9/tensor_names#save_3/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_3/Assign_9AssignDefault/first_layer/filterssave_3/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_3/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_10	RestoreV2save_3/Const save_3/RestoreV2_10/tensor_names$save_3/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_3/Assign_10Assign Default/first_layer/filters/Adamsave_3/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_3/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_11	RestoreV2save_3/Const save_3/RestoreV2_11/tensor_names$save_3/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_3/Assign_11Assign"Default/first_layer/filters/Adam_1save_3/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_3/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_12	RestoreV2save_3/Const save_3/RestoreV2_12/tensor_names$save_3/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_3/Assign_12AssignDefault/last_layer/biasessave_3/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_3/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_13	RestoreV2save_3/Const save_3/RestoreV2_13/tensor_names$save_3/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_3/Assign_13AssignDefault/last_layer/biases/Adamsave_3/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_3/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_14	RestoreV2save_3/Const save_3/RestoreV2_14/tensor_names$save_3/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_3/Assign_14Assign Default/last_layer/biases/Adam_1save_3/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_3/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_15	RestoreV2save_3/Const save_3/RestoreV2_15/tensor_names$save_3/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_3/Assign_15AssignDefault/last_layer/filterssave_3/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_3/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_16	RestoreV2save_3/Const save_3/RestoreV2_16/tensor_names$save_3/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_3/Assign_16AssignDefault/last_layer/filters/Adamsave_3/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_3/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_17	RestoreV2save_3/Const save_3/RestoreV2_17/tensor_names$save_3/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_3/Assign_17Assign!Default/last_layer/filters/Adam_1save_3/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_3/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_18	RestoreV2save_3/Const save_3/RestoreV2_18/tensor_names$save_3/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_18Assign$Default/residual_block0/conv1/biasessave_3/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_3/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_19	RestoreV2save_3/Const save_3/RestoreV2_19/tensor_names$save_3/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_3/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_3/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_20	RestoreV2save_3/Const save_3/RestoreV2_20/tensor_names$save_3/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_3/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_3/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_21	RestoreV2save_3/Const save_3/RestoreV2_21/tensor_names$save_3/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_21Assign%Default/residual_block0/conv1/filterssave_3/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_3/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_22	RestoreV2save_3/Const save_3/RestoreV2_22/tensor_names$save_3/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_3/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_3/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_23	RestoreV2save_3/Const save_3/RestoreV2_23/tensor_names$save_3/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_3/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_3/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_24	RestoreV2save_3/Const save_3/RestoreV2_24/tensor_names$save_3/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_24Assign$Default/residual_block0/conv2/biasessave_3/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_3/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_25	RestoreV2save_3/Const save_3/RestoreV2_25/tensor_names$save_3/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_3/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_3/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_26	RestoreV2save_3/Const save_3/RestoreV2_26/tensor_names$save_3/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_3/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_3/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_27	RestoreV2save_3/Const save_3/RestoreV2_27/tensor_names$save_3/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_27Assign%Default/residual_block0/conv2/filterssave_3/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_3/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_28	RestoreV2save_3/Const save_3/RestoreV2_28/tensor_names$save_3/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_3/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_3/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_29	RestoreV2save_3/Const save_3/RestoreV2_29/tensor_names$save_3/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_3/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_3/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_30	RestoreV2save_3/Const save_3/RestoreV2_30/tensor_names$save_3/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_30Assign$Default/residual_block1/conv1/biasessave_3/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_3/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_31	RestoreV2save_3/Const save_3/RestoreV2_31/tensor_names$save_3/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_3/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_3/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_32	RestoreV2save_3/Const save_3/RestoreV2_32/tensor_names$save_3/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_3/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_3/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_33	RestoreV2save_3/Const save_3/RestoreV2_33/tensor_names$save_3/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_33Assign%Default/residual_block1/conv1/filterssave_3/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_3/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_34	RestoreV2save_3/Const save_3/RestoreV2_34/tensor_names$save_3/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_3/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_3/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_35	RestoreV2save_3/Const save_3/RestoreV2_35/tensor_names$save_3/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_3/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_3/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_36	RestoreV2save_3/Const save_3/RestoreV2_36/tensor_names$save_3/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_36Assign$Default/residual_block1/conv2/biasessave_3/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_3/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_37	RestoreV2save_3/Const save_3/RestoreV2_37/tensor_names$save_3/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_3/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_3/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_38	RestoreV2save_3/Const save_3/RestoreV2_38/tensor_names$save_3/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_3/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_3/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_39	RestoreV2save_3/Const save_3/RestoreV2_39/tensor_names$save_3/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_39Assign%Default/residual_block1/conv2/filterssave_3/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_3/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_40	RestoreV2save_3/Const save_3/RestoreV2_40/tensor_names$save_3/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_3/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_3/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_41	RestoreV2save_3/Const save_3/RestoreV2_41/tensor_names$save_3/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_3/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_3/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_42	RestoreV2save_3/Const save_3/RestoreV2_42/tensor_names$save_3/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_42Assign$Default/residual_block2/conv1/biasessave_3/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_3/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_43	RestoreV2save_3/Const save_3/RestoreV2_43/tensor_names$save_3/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_3/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_3/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_44	RestoreV2save_3/Const save_3/RestoreV2_44/tensor_names$save_3/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_3/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_3/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_45	RestoreV2save_3/Const save_3/RestoreV2_45/tensor_names$save_3/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_45Assign%Default/residual_block2/conv1/filterssave_3/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_3/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_46	RestoreV2save_3/Const save_3/RestoreV2_46/tensor_names$save_3/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_3/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_3/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_47	RestoreV2save_3/Const save_3/RestoreV2_47/tensor_names$save_3/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_3/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_3/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_48	RestoreV2save_3/Const save_3/RestoreV2_48/tensor_names$save_3/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_48Assign$Default/residual_block2/conv2/biasessave_3/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_3/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_49	RestoreV2save_3/Const save_3/RestoreV2_49/tensor_names$save_3/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_3/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_3/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_50	RestoreV2save_3/Const save_3/RestoreV2_50/tensor_names$save_3/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_3/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_3/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_51	RestoreV2save_3/Const save_3/RestoreV2_51/tensor_names$save_3/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_51Assign%Default/residual_block2/conv2/filterssave_3/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_3/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_52	RestoreV2save_3/Const save_3/RestoreV2_52/tensor_names$save_3/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_3/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_3/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_53	RestoreV2save_3/Const save_3/RestoreV2_53/tensor_names$save_3/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_3/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_3/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_54	RestoreV2save_3/Const save_3/RestoreV2_54/tensor_names$save_3/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_54Assign$Default/residual_block3/conv1/biasessave_3/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_3/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_55	RestoreV2save_3/Const save_3/RestoreV2_55/tensor_names$save_3/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_3/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_3/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_56	RestoreV2save_3/Const save_3/RestoreV2_56/tensor_names$save_3/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_3/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_3/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_57	RestoreV2save_3/Const save_3/RestoreV2_57/tensor_names$save_3/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_57Assign%Default/residual_block3/conv1/filterssave_3/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_3/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_58	RestoreV2save_3/Const save_3/RestoreV2_58/tensor_names$save_3/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_3/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_3/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_59	RestoreV2save_3/Const save_3/RestoreV2_59/tensor_names$save_3/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_3/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_3/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_60	RestoreV2save_3/Const save_3/RestoreV2_60/tensor_names$save_3/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_60Assign$Default/residual_block3/conv2/biasessave_3/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_3/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_61	RestoreV2save_3/Const save_3/RestoreV2_61/tensor_names$save_3/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_3/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_3/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_62	RestoreV2save_3/Const save_3/RestoreV2_62/tensor_names$save_3/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_3/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_3/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_63	RestoreV2save_3/Const save_3/RestoreV2_63/tensor_names$save_3/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_63Assign%Default/residual_block3/conv2/filterssave_3/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_3/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_64	RestoreV2save_3/Const save_3/RestoreV2_64/tensor_names$save_3/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_3/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_3/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_65	RestoreV2save_3/Const save_3/RestoreV2_65/tensor_names$save_3/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_3/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_3/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_66	RestoreV2save_3/Const save_3/RestoreV2_66/tensor_names$save_3/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_66Assign$Default/residual_block4/conv1/biasessave_3/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_3/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_67	RestoreV2save_3/Const save_3/RestoreV2_67/tensor_names$save_3/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_3/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_3/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_68	RestoreV2save_3/Const save_3/RestoreV2_68/tensor_names$save_3/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_3/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_3/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_69	RestoreV2save_3/Const save_3/RestoreV2_69/tensor_names$save_3/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_69Assign%Default/residual_block4/conv1/filterssave_3/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_3/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_70	RestoreV2save_3/Const save_3/RestoreV2_70/tensor_names$save_3/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_3/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_3/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_71	RestoreV2save_3/Const save_3/RestoreV2_71/tensor_names$save_3/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_3/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_3/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_72	RestoreV2save_3/Const save_3/RestoreV2_72/tensor_names$save_3/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_3/Assign_72Assign$Default/residual_block4/conv2/biasessave_3/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_3/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_73	RestoreV2save_3/Const save_3/RestoreV2_73/tensor_names$save_3/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_3/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_3/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_3/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_74	RestoreV2save_3/Const save_3/RestoreV2_74/tensor_names$save_3/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_3/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_3/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_3/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_75	RestoreV2save_3/Const save_3/RestoreV2_75/tensor_names$save_3/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_3/Assign_75Assign%Default/residual_block4/conv2/filterssave_3/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_3/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_76	RestoreV2save_3/Const save_3/RestoreV2_76/tensor_names$save_3/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_3/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_3/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_3/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_77	RestoreV2save_3/Const save_3/RestoreV2_77/tensor_names$save_3/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_3/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_3/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_3/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_78	RestoreV2save_3/Const save_3/RestoreV2_78/tensor_names$save_3/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_3/Assign_78Assignbeta1_powersave_3/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_3/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_3/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_3/RestoreV2_79	RestoreV2save_3/Const save_3/RestoreV2_79/tensor_names$save_3/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_3/Assign_79Assignbeta2_powersave_3/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_2^save_3/Assign_3^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_68^save_3/Assign_69^save_3/Assign_70^save_3/Assign_71^save_3/Assign_72^save_3/Assign_73^save_3/Assign_74^save_3/Assign_75^save_3/Assign_76^save_3/Assign_77^save_3/Assign_78^save_3/Assign_79
1
save_3/restore_allNoOp^save_3/restore_shard
R
save_4/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_ce8acf2d7a56433da25c67088369cc24/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
ē
save_4/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_4/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
£
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/control_dependency^save_4/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_4/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_4/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_4/AssignAssign"Default/down_sampling_layer/biasessave_4/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_4/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_1	RestoreV2save_4/Constsave_4/RestoreV2_1/tensor_names#save_4/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_4/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_4/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_4/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_2	RestoreV2save_4/Constsave_4/RestoreV2_2/tensor_names#save_4/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_4/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_4/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_4/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_3	RestoreV2save_4/Constsave_4/RestoreV2_3/tensor_names#save_4/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_4/Assign_3Assign#Default/down_sampling_layer/filterssave_4/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_4/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_4	RestoreV2save_4/Constsave_4/RestoreV2_4/tensor_names#save_4/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_4/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_4/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_4/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_5	RestoreV2save_4/Constsave_4/RestoreV2_5/tensor_names#save_4/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_4/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_4/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_4/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_6	RestoreV2save_4/Constsave_4/RestoreV2_6/tensor_names#save_4/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_4/Assign_6AssignDefault/first_layer/biasessave_4/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_4/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_7	RestoreV2save_4/Constsave_4/RestoreV2_7/tensor_names#save_4/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_4/Assign_7AssignDefault/first_layer/biases/Adamsave_4/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_4/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_8	RestoreV2save_4/Constsave_4/RestoreV2_8/tensor_names#save_4/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_4/Assign_8Assign!Default/first_layer/biases/Adam_1save_4/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_4/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_4/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_4/RestoreV2_9	RestoreV2save_4/Constsave_4/RestoreV2_9/tensor_names#save_4/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_4/Assign_9AssignDefault/first_layer/filterssave_4/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_4/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_10	RestoreV2save_4/Const save_4/RestoreV2_10/tensor_names$save_4/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_4/Assign_10Assign Default/first_layer/filters/Adamsave_4/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_4/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_11	RestoreV2save_4/Const save_4/RestoreV2_11/tensor_names$save_4/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_4/Assign_11Assign"Default/first_layer/filters/Adam_1save_4/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_4/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_12	RestoreV2save_4/Const save_4/RestoreV2_12/tensor_names$save_4/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_4/Assign_12AssignDefault/last_layer/biasessave_4/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_4/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_13	RestoreV2save_4/Const save_4/RestoreV2_13/tensor_names$save_4/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_4/Assign_13AssignDefault/last_layer/biases/Adamsave_4/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_4/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_14	RestoreV2save_4/Const save_4/RestoreV2_14/tensor_names$save_4/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_4/Assign_14Assign Default/last_layer/biases/Adam_1save_4/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_4/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_15	RestoreV2save_4/Const save_4/RestoreV2_15/tensor_names$save_4/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_4/Assign_15AssignDefault/last_layer/filterssave_4/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_4/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_16	RestoreV2save_4/Const save_4/RestoreV2_16/tensor_names$save_4/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_4/Assign_16AssignDefault/last_layer/filters/Adamsave_4/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_4/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_17	RestoreV2save_4/Const save_4/RestoreV2_17/tensor_names$save_4/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_4/Assign_17Assign!Default/last_layer/filters/Adam_1save_4/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_4/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_18	RestoreV2save_4/Const save_4/RestoreV2_18/tensor_names$save_4/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_18Assign$Default/residual_block0/conv1/biasessave_4/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_4/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_19	RestoreV2save_4/Const save_4/RestoreV2_19/tensor_names$save_4/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_4/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_4/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_20	RestoreV2save_4/Const save_4/RestoreV2_20/tensor_names$save_4/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_4/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_4/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_21	RestoreV2save_4/Const save_4/RestoreV2_21/tensor_names$save_4/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_21Assign%Default/residual_block0/conv1/filterssave_4/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_4/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_22	RestoreV2save_4/Const save_4/RestoreV2_22/tensor_names$save_4/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_4/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_4/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_23	RestoreV2save_4/Const save_4/RestoreV2_23/tensor_names$save_4/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_4/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_4/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_24	RestoreV2save_4/Const save_4/RestoreV2_24/tensor_names$save_4/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_24Assign$Default/residual_block0/conv2/biasessave_4/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_4/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_25	RestoreV2save_4/Const save_4/RestoreV2_25/tensor_names$save_4/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_4/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_4/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_26	RestoreV2save_4/Const save_4/RestoreV2_26/tensor_names$save_4/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_4/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_4/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_27	RestoreV2save_4/Const save_4/RestoreV2_27/tensor_names$save_4/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_27Assign%Default/residual_block0/conv2/filterssave_4/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_4/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_28	RestoreV2save_4/Const save_4/RestoreV2_28/tensor_names$save_4/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_4/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_4/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_29	RestoreV2save_4/Const save_4/RestoreV2_29/tensor_names$save_4/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_4/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_4/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_30	RestoreV2save_4/Const save_4/RestoreV2_30/tensor_names$save_4/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_30Assign$Default/residual_block1/conv1/biasessave_4/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_4/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_31	RestoreV2save_4/Const save_4/RestoreV2_31/tensor_names$save_4/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_4/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_4/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_32	RestoreV2save_4/Const save_4/RestoreV2_32/tensor_names$save_4/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_4/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_4/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_33	RestoreV2save_4/Const save_4/RestoreV2_33/tensor_names$save_4/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_33Assign%Default/residual_block1/conv1/filterssave_4/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_4/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_34	RestoreV2save_4/Const save_4/RestoreV2_34/tensor_names$save_4/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_4/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_4/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_35	RestoreV2save_4/Const save_4/RestoreV2_35/tensor_names$save_4/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_4/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_4/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_36	RestoreV2save_4/Const save_4/RestoreV2_36/tensor_names$save_4/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_36Assign$Default/residual_block1/conv2/biasessave_4/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_4/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_37	RestoreV2save_4/Const save_4/RestoreV2_37/tensor_names$save_4/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_4/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_4/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_38	RestoreV2save_4/Const save_4/RestoreV2_38/tensor_names$save_4/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_4/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_4/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_39	RestoreV2save_4/Const save_4/RestoreV2_39/tensor_names$save_4/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_39Assign%Default/residual_block1/conv2/filterssave_4/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_4/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_40	RestoreV2save_4/Const save_4/RestoreV2_40/tensor_names$save_4/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_4/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_4/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_41	RestoreV2save_4/Const save_4/RestoreV2_41/tensor_names$save_4/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_4/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_4/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_42	RestoreV2save_4/Const save_4/RestoreV2_42/tensor_names$save_4/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_42Assign$Default/residual_block2/conv1/biasessave_4/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_4/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_43	RestoreV2save_4/Const save_4/RestoreV2_43/tensor_names$save_4/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_4/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_4/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_44	RestoreV2save_4/Const save_4/RestoreV2_44/tensor_names$save_4/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_4/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_4/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_45	RestoreV2save_4/Const save_4/RestoreV2_45/tensor_names$save_4/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_45Assign%Default/residual_block2/conv1/filterssave_4/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_4/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_46	RestoreV2save_4/Const save_4/RestoreV2_46/tensor_names$save_4/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_4/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_4/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_47	RestoreV2save_4/Const save_4/RestoreV2_47/tensor_names$save_4/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_4/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_4/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_48	RestoreV2save_4/Const save_4/RestoreV2_48/tensor_names$save_4/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_48Assign$Default/residual_block2/conv2/biasessave_4/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_4/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_49	RestoreV2save_4/Const save_4/RestoreV2_49/tensor_names$save_4/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_4/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_4/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_50	RestoreV2save_4/Const save_4/RestoreV2_50/tensor_names$save_4/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_4/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_4/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_51	RestoreV2save_4/Const save_4/RestoreV2_51/tensor_names$save_4/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_51Assign%Default/residual_block2/conv2/filterssave_4/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_4/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_52	RestoreV2save_4/Const save_4/RestoreV2_52/tensor_names$save_4/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_4/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_4/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_53	RestoreV2save_4/Const save_4/RestoreV2_53/tensor_names$save_4/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_4/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_4/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_54	RestoreV2save_4/Const save_4/RestoreV2_54/tensor_names$save_4/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_54Assign$Default/residual_block3/conv1/biasessave_4/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_4/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_55	RestoreV2save_4/Const save_4/RestoreV2_55/tensor_names$save_4/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_4/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_4/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_56	RestoreV2save_4/Const save_4/RestoreV2_56/tensor_names$save_4/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_4/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_4/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_57	RestoreV2save_4/Const save_4/RestoreV2_57/tensor_names$save_4/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_57Assign%Default/residual_block3/conv1/filterssave_4/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_4/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_58	RestoreV2save_4/Const save_4/RestoreV2_58/tensor_names$save_4/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_4/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_4/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_59	RestoreV2save_4/Const save_4/RestoreV2_59/tensor_names$save_4/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_4/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_4/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_60	RestoreV2save_4/Const save_4/RestoreV2_60/tensor_names$save_4/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_60Assign$Default/residual_block3/conv2/biasessave_4/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_4/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_61	RestoreV2save_4/Const save_4/RestoreV2_61/tensor_names$save_4/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_4/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_4/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_62	RestoreV2save_4/Const save_4/RestoreV2_62/tensor_names$save_4/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_4/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_4/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_63	RestoreV2save_4/Const save_4/RestoreV2_63/tensor_names$save_4/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_63Assign%Default/residual_block3/conv2/filterssave_4/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_4/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_64	RestoreV2save_4/Const save_4/RestoreV2_64/tensor_names$save_4/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_4/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_4/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_65	RestoreV2save_4/Const save_4/RestoreV2_65/tensor_names$save_4/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_4/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_4/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_66	RestoreV2save_4/Const save_4/RestoreV2_66/tensor_names$save_4/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_66Assign$Default/residual_block4/conv1/biasessave_4/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_4/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_67	RestoreV2save_4/Const save_4/RestoreV2_67/tensor_names$save_4/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_4/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_4/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_68	RestoreV2save_4/Const save_4/RestoreV2_68/tensor_names$save_4/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_4/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_4/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_69	RestoreV2save_4/Const save_4/RestoreV2_69/tensor_names$save_4/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_69Assign%Default/residual_block4/conv1/filterssave_4/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_4/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_70	RestoreV2save_4/Const save_4/RestoreV2_70/tensor_names$save_4/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_4/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_4/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_71	RestoreV2save_4/Const save_4/RestoreV2_71/tensor_names$save_4/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_4/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_4/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_72	RestoreV2save_4/Const save_4/RestoreV2_72/tensor_names$save_4/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_4/Assign_72Assign$Default/residual_block4/conv2/biasessave_4/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_4/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_73	RestoreV2save_4/Const save_4/RestoreV2_73/tensor_names$save_4/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_4/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_4/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_4/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_74	RestoreV2save_4/Const save_4/RestoreV2_74/tensor_names$save_4/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_4/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_4/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_4/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_75	RestoreV2save_4/Const save_4/RestoreV2_75/tensor_names$save_4/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_4/Assign_75Assign%Default/residual_block4/conv2/filterssave_4/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_4/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_76	RestoreV2save_4/Const save_4/RestoreV2_76/tensor_names$save_4/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_4/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_4/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_4/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_77	RestoreV2save_4/Const save_4/RestoreV2_77/tensor_names$save_4/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_4/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_4/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_4/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_78	RestoreV2save_4/Const save_4/RestoreV2_78/tensor_names$save_4/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_4/Assign_78Assignbeta1_powersave_4/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_4/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_4/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_4/RestoreV2_79	RestoreV2save_4/Const save_4/RestoreV2_79/tensor_names$save_4/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_4/Assign_79Assignbeta2_powersave_4/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_2^save_4/Assign_3^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_50^save_4/Assign_51^save_4/Assign_52^save_4/Assign_53^save_4/Assign_54^save_4/Assign_55^save_4/Assign_56^save_4/Assign_57^save_4/Assign_58^save_4/Assign_59^save_4/Assign_60^save_4/Assign_61^save_4/Assign_62^save_4/Assign_63^save_4/Assign_64^save_4/Assign_65^save_4/Assign_66^save_4/Assign_67^save_4/Assign_68^save_4/Assign_69^save_4/Assign_70^save_4/Assign_71^save_4/Assign_72^save_4/Assign_73^save_4/Assign_74^save_4/Assign_75^save_4/Assign_76^save_4/Assign_77^save_4/Assign_78^save_4/Assign_79
1
save_4/restore_allNoOp^save_4/restore_shard
R
save_5/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_7f936a8b32ea450699d4249c800c501b/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
ē
save_5/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_5/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: 
£
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/control_dependency^save_5/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_5/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_5/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_5/AssignAssign"Default/down_sampling_layer/biasessave_5/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_5/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_1	RestoreV2save_5/Constsave_5/RestoreV2_1/tensor_names#save_5/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_5/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_5/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_5/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_2	RestoreV2save_5/Constsave_5/RestoreV2_2/tensor_names#save_5/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_5/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_5/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_5/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_3	RestoreV2save_5/Constsave_5/RestoreV2_3/tensor_names#save_5/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_5/Assign_3Assign#Default/down_sampling_layer/filterssave_5/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_5/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_4	RestoreV2save_5/Constsave_5/RestoreV2_4/tensor_names#save_5/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_5/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_5/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_5/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_5	RestoreV2save_5/Constsave_5/RestoreV2_5/tensor_names#save_5/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_5/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_5/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_5/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_6	RestoreV2save_5/Constsave_5/RestoreV2_6/tensor_names#save_5/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_5/Assign_6AssignDefault/first_layer/biasessave_5/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_5/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_7	RestoreV2save_5/Constsave_5/RestoreV2_7/tensor_names#save_5/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_5/Assign_7AssignDefault/first_layer/biases/Adamsave_5/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_5/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_8	RestoreV2save_5/Constsave_5/RestoreV2_8/tensor_names#save_5/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_5/Assign_8Assign!Default/first_layer/biases/Adam_1save_5/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_5/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_5/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_5/RestoreV2_9	RestoreV2save_5/Constsave_5/RestoreV2_9/tensor_names#save_5/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_5/Assign_9AssignDefault/first_layer/filterssave_5/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_5/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_10	RestoreV2save_5/Const save_5/RestoreV2_10/tensor_names$save_5/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_5/Assign_10Assign Default/first_layer/filters/Adamsave_5/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_5/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_11	RestoreV2save_5/Const save_5/RestoreV2_11/tensor_names$save_5/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_5/Assign_11Assign"Default/first_layer/filters/Adam_1save_5/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_5/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_12	RestoreV2save_5/Const save_5/RestoreV2_12/tensor_names$save_5/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_5/Assign_12AssignDefault/last_layer/biasessave_5/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_5/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_13	RestoreV2save_5/Const save_5/RestoreV2_13/tensor_names$save_5/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_5/Assign_13AssignDefault/last_layer/biases/Adamsave_5/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_5/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_14	RestoreV2save_5/Const save_5/RestoreV2_14/tensor_names$save_5/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_5/Assign_14Assign Default/last_layer/biases/Adam_1save_5/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_5/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_15	RestoreV2save_5/Const save_5/RestoreV2_15/tensor_names$save_5/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_5/Assign_15AssignDefault/last_layer/filterssave_5/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_5/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_16	RestoreV2save_5/Const save_5/RestoreV2_16/tensor_names$save_5/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_5/Assign_16AssignDefault/last_layer/filters/Adamsave_5/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_5/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_17	RestoreV2save_5/Const save_5/RestoreV2_17/tensor_names$save_5/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_5/Assign_17Assign!Default/last_layer/filters/Adam_1save_5/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_5/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_18	RestoreV2save_5/Const save_5/RestoreV2_18/tensor_names$save_5/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_18Assign$Default/residual_block0/conv1/biasessave_5/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_5/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_19	RestoreV2save_5/Const save_5/RestoreV2_19/tensor_names$save_5/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_5/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_5/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_20	RestoreV2save_5/Const save_5/RestoreV2_20/tensor_names$save_5/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_5/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_5/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_21	RestoreV2save_5/Const save_5/RestoreV2_21/tensor_names$save_5/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_21Assign%Default/residual_block0/conv1/filterssave_5/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_5/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_22	RestoreV2save_5/Const save_5/RestoreV2_22/tensor_names$save_5/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_5/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_5/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_23	RestoreV2save_5/Const save_5/RestoreV2_23/tensor_names$save_5/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_5/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_5/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_24	RestoreV2save_5/Const save_5/RestoreV2_24/tensor_names$save_5/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_24Assign$Default/residual_block0/conv2/biasessave_5/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_5/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_25	RestoreV2save_5/Const save_5/RestoreV2_25/tensor_names$save_5/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_5/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_5/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_26	RestoreV2save_5/Const save_5/RestoreV2_26/tensor_names$save_5/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_5/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_5/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_27	RestoreV2save_5/Const save_5/RestoreV2_27/tensor_names$save_5/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_27Assign%Default/residual_block0/conv2/filterssave_5/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_5/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_28	RestoreV2save_5/Const save_5/RestoreV2_28/tensor_names$save_5/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_5/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_5/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_29	RestoreV2save_5/Const save_5/RestoreV2_29/tensor_names$save_5/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_5/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_5/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_30	RestoreV2save_5/Const save_5/RestoreV2_30/tensor_names$save_5/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_30Assign$Default/residual_block1/conv1/biasessave_5/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_5/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_31	RestoreV2save_5/Const save_5/RestoreV2_31/tensor_names$save_5/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_5/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_5/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_32	RestoreV2save_5/Const save_5/RestoreV2_32/tensor_names$save_5/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_5/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_5/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_33	RestoreV2save_5/Const save_5/RestoreV2_33/tensor_names$save_5/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_33Assign%Default/residual_block1/conv1/filterssave_5/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_5/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_34	RestoreV2save_5/Const save_5/RestoreV2_34/tensor_names$save_5/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_5/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_5/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_35	RestoreV2save_5/Const save_5/RestoreV2_35/tensor_names$save_5/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_5/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_5/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_36	RestoreV2save_5/Const save_5/RestoreV2_36/tensor_names$save_5/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_36Assign$Default/residual_block1/conv2/biasessave_5/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_5/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_37	RestoreV2save_5/Const save_5/RestoreV2_37/tensor_names$save_5/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_5/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_5/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_38	RestoreV2save_5/Const save_5/RestoreV2_38/tensor_names$save_5/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_5/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_5/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_39	RestoreV2save_5/Const save_5/RestoreV2_39/tensor_names$save_5/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_39Assign%Default/residual_block1/conv2/filterssave_5/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_5/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_40	RestoreV2save_5/Const save_5/RestoreV2_40/tensor_names$save_5/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_5/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_5/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_41	RestoreV2save_5/Const save_5/RestoreV2_41/tensor_names$save_5/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_5/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_5/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_42	RestoreV2save_5/Const save_5/RestoreV2_42/tensor_names$save_5/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_42Assign$Default/residual_block2/conv1/biasessave_5/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_5/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_43	RestoreV2save_5/Const save_5/RestoreV2_43/tensor_names$save_5/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_5/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_5/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_44	RestoreV2save_5/Const save_5/RestoreV2_44/tensor_names$save_5/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_5/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_5/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_45	RestoreV2save_5/Const save_5/RestoreV2_45/tensor_names$save_5/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_45Assign%Default/residual_block2/conv1/filterssave_5/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_5/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_46	RestoreV2save_5/Const save_5/RestoreV2_46/tensor_names$save_5/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_5/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_5/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_47	RestoreV2save_5/Const save_5/RestoreV2_47/tensor_names$save_5/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_5/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_5/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_48	RestoreV2save_5/Const save_5/RestoreV2_48/tensor_names$save_5/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_48Assign$Default/residual_block2/conv2/biasessave_5/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_5/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_49	RestoreV2save_5/Const save_5/RestoreV2_49/tensor_names$save_5/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_5/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_5/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_50	RestoreV2save_5/Const save_5/RestoreV2_50/tensor_names$save_5/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_5/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_5/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_51	RestoreV2save_5/Const save_5/RestoreV2_51/tensor_names$save_5/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_51Assign%Default/residual_block2/conv2/filterssave_5/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_5/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_52	RestoreV2save_5/Const save_5/RestoreV2_52/tensor_names$save_5/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_5/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_5/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_53	RestoreV2save_5/Const save_5/RestoreV2_53/tensor_names$save_5/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_5/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_5/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_54	RestoreV2save_5/Const save_5/RestoreV2_54/tensor_names$save_5/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_54Assign$Default/residual_block3/conv1/biasessave_5/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_5/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_55	RestoreV2save_5/Const save_5/RestoreV2_55/tensor_names$save_5/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_5/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_5/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_56	RestoreV2save_5/Const save_5/RestoreV2_56/tensor_names$save_5/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_5/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_5/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_57	RestoreV2save_5/Const save_5/RestoreV2_57/tensor_names$save_5/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_57Assign%Default/residual_block3/conv1/filterssave_5/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_5/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_58	RestoreV2save_5/Const save_5/RestoreV2_58/tensor_names$save_5/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_5/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_5/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_59	RestoreV2save_5/Const save_5/RestoreV2_59/tensor_names$save_5/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_5/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_5/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_60	RestoreV2save_5/Const save_5/RestoreV2_60/tensor_names$save_5/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_60Assign$Default/residual_block3/conv2/biasessave_5/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_5/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_61	RestoreV2save_5/Const save_5/RestoreV2_61/tensor_names$save_5/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_5/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_5/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_62	RestoreV2save_5/Const save_5/RestoreV2_62/tensor_names$save_5/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_5/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_5/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_63	RestoreV2save_5/Const save_5/RestoreV2_63/tensor_names$save_5/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_63Assign%Default/residual_block3/conv2/filterssave_5/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_5/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_64	RestoreV2save_5/Const save_5/RestoreV2_64/tensor_names$save_5/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_5/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_5/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_65	RestoreV2save_5/Const save_5/RestoreV2_65/tensor_names$save_5/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_5/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_5/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_66	RestoreV2save_5/Const save_5/RestoreV2_66/tensor_names$save_5/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_66Assign$Default/residual_block4/conv1/biasessave_5/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_5/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_67	RestoreV2save_5/Const save_5/RestoreV2_67/tensor_names$save_5/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_5/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_5/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_68	RestoreV2save_5/Const save_5/RestoreV2_68/tensor_names$save_5/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_5/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_5/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_69	RestoreV2save_5/Const save_5/RestoreV2_69/tensor_names$save_5/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_69Assign%Default/residual_block4/conv1/filterssave_5/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_5/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_70	RestoreV2save_5/Const save_5/RestoreV2_70/tensor_names$save_5/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_5/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_5/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_71	RestoreV2save_5/Const save_5/RestoreV2_71/tensor_names$save_5/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_5/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_5/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_72	RestoreV2save_5/Const save_5/RestoreV2_72/tensor_names$save_5/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_5/Assign_72Assign$Default/residual_block4/conv2/biasessave_5/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_5/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_73	RestoreV2save_5/Const save_5/RestoreV2_73/tensor_names$save_5/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_5/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_5/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_5/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_74	RestoreV2save_5/Const save_5/RestoreV2_74/tensor_names$save_5/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_5/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_5/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_5/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_75	RestoreV2save_5/Const save_5/RestoreV2_75/tensor_names$save_5/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_5/Assign_75Assign%Default/residual_block4/conv2/filterssave_5/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_5/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_76	RestoreV2save_5/Const save_5/RestoreV2_76/tensor_names$save_5/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_5/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_5/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_5/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_77	RestoreV2save_5/Const save_5/RestoreV2_77/tensor_names$save_5/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_5/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_5/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_5/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_78	RestoreV2save_5/Const save_5/RestoreV2_78/tensor_names$save_5/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_5/Assign_78Assignbeta1_powersave_5/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_5/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_5/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_5/RestoreV2_79	RestoreV2save_5/Const save_5/RestoreV2_79/tensor_names$save_5/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_5/Assign_79Assignbeta2_powersave_5/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_2^save_5/Assign_3^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_50^save_5/Assign_51^save_5/Assign_52^save_5/Assign_53^save_5/Assign_54^save_5/Assign_55^save_5/Assign_56^save_5/Assign_57^save_5/Assign_58^save_5/Assign_59^save_5/Assign_60^save_5/Assign_61^save_5/Assign_62^save_5/Assign_63^save_5/Assign_64^save_5/Assign_65^save_5/Assign_66^save_5/Assign_67^save_5/Assign_68^save_5/Assign_69^save_5/Assign_70^save_5/Assign_71^save_5/Assign_72^save_5/Assign_73^save_5/Assign_74^save_5/Assign_75^save_5/Assign_76^save_5/Assign_77^save_5/Assign_78^save_5/Assign_79
1
save_5/restore_allNoOp^save_5/restore_shard
R
save_6/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_f6ba4b8520024431b2939e699331132c/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
ē
save_6/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_6/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
£
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/control_dependency^save_6/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_6/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_6/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_6/AssignAssign"Default/down_sampling_layer/biasessave_6/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_6/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_1	RestoreV2save_6/Constsave_6/RestoreV2_1/tensor_names#save_6/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_6/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_6/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_6/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_2	RestoreV2save_6/Constsave_6/RestoreV2_2/tensor_names#save_6/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_6/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_6/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_6/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_3	RestoreV2save_6/Constsave_6/RestoreV2_3/tensor_names#save_6/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_6/Assign_3Assign#Default/down_sampling_layer/filterssave_6/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_6/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_4	RestoreV2save_6/Constsave_6/RestoreV2_4/tensor_names#save_6/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_6/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_6/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_6/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_5	RestoreV2save_6/Constsave_6/RestoreV2_5/tensor_names#save_6/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_6/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_6/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_6/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_6	RestoreV2save_6/Constsave_6/RestoreV2_6/tensor_names#save_6/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_6/Assign_6AssignDefault/first_layer/biasessave_6/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_6/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_7	RestoreV2save_6/Constsave_6/RestoreV2_7/tensor_names#save_6/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_6/Assign_7AssignDefault/first_layer/biases/Adamsave_6/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_6/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_8	RestoreV2save_6/Constsave_6/RestoreV2_8/tensor_names#save_6/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_6/Assign_8Assign!Default/first_layer/biases/Adam_1save_6/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_6/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_6/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_6/RestoreV2_9	RestoreV2save_6/Constsave_6/RestoreV2_9/tensor_names#save_6/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_6/Assign_9AssignDefault/first_layer/filterssave_6/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_6/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_10	RestoreV2save_6/Const save_6/RestoreV2_10/tensor_names$save_6/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_6/Assign_10Assign Default/first_layer/filters/Adamsave_6/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_6/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_11	RestoreV2save_6/Const save_6/RestoreV2_11/tensor_names$save_6/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_6/Assign_11Assign"Default/first_layer/filters/Adam_1save_6/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_6/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_12	RestoreV2save_6/Const save_6/RestoreV2_12/tensor_names$save_6/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_6/Assign_12AssignDefault/last_layer/biasessave_6/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_6/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_13	RestoreV2save_6/Const save_6/RestoreV2_13/tensor_names$save_6/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_6/Assign_13AssignDefault/last_layer/biases/Adamsave_6/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_6/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_14	RestoreV2save_6/Const save_6/RestoreV2_14/tensor_names$save_6/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_6/Assign_14Assign Default/last_layer/biases/Adam_1save_6/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_6/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_15	RestoreV2save_6/Const save_6/RestoreV2_15/tensor_names$save_6/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_6/Assign_15AssignDefault/last_layer/filterssave_6/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_6/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_16	RestoreV2save_6/Const save_6/RestoreV2_16/tensor_names$save_6/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_6/Assign_16AssignDefault/last_layer/filters/Adamsave_6/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_6/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_17	RestoreV2save_6/Const save_6/RestoreV2_17/tensor_names$save_6/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_6/Assign_17Assign!Default/last_layer/filters/Adam_1save_6/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_6/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_18	RestoreV2save_6/Const save_6/RestoreV2_18/tensor_names$save_6/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_18Assign$Default/residual_block0/conv1/biasessave_6/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_6/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_19	RestoreV2save_6/Const save_6/RestoreV2_19/tensor_names$save_6/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_6/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_6/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_20	RestoreV2save_6/Const save_6/RestoreV2_20/tensor_names$save_6/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_6/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_6/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_21	RestoreV2save_6/Const save_6/RestoreV2_21/tensor_names$save_6/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_21Assign%Default/residual_block0/conv1/filterssave_6/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_6/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_22	RestoreV2save_6/Const save_6/RestoreV2_22/tensor_names$save_6/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_6/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_6/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_23	RestoreV2save_6/Const save_6/RestoreV2_23/tensor_names$save_6/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_6/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_6/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_24	RestoreV2save_6/Const save_6/RestoreV2_24/tensor_names$save_6/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_24Assign$Default/residual_block0/conv2/biasessave_6/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_6/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_25	RestoreV2save_6/Const save_6/RestoreV2_25/tensor_names$save_6/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_6/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_6/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_26	RestoreV2save_6/Const save_6/RestoreV2_26/tensor_names$save_6/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_6/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_6/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_27	RestoreV2save_6/Const save_6/RestoreV2_27/tensor_names$save_6/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_27Assign%Default/residual_block0/conv2/filterssave_6/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_6/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_28	RestoreV2save_6/Const save_6/RestoreV2_28/tensor_names$save_6/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_6/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_6/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_29	RestoreV2save_6/Const save_6/RestoreV2_29/tensor_names$save_6/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_6/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_6/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_30	RestoreV2save_6/Const save_6/RestoreV2_30/tensor_names$save_6/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_30Assign$Default/residual_block1/conv1/biasessave_6/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_6/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_31	RestoreV2save_6/Const save_6/RestoreV2_31/tensor_names$save_6/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_6/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_6/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_32	RestoreV2save_6/Const save_6/RestoreV2_32/tensor_names$save_6/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_6/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_6/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_33	RestoreV2save_6/Const save_6/RestoreV2_33/tensor_names$save_6/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_33Assign%Default/residual_block1/conv1/filterssave_6/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_6/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_34	RestoreV2save_6/Const save_6/RestoreV2_34/tensor_names$save_6/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_6/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_6/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_35	RestoreV2save_6/Const save_6/RestoreV2_35/tensor_names$save_6/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_6/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_6/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_36	RestoreV2save_6/Const save_6/RestoreV2_36/tensor_names$save_6/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_36Assign$Default/residual_block1/conv2/biasessave_6/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_6/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_37	RestoreV2save_6/Const save_6/RestoreV2_37/tensor_names$save_6/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_6/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_6/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_38	RestoreV2save_6/Const save_6/RestoreV2_38/tensor_names$save_6/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_6/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_6/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_39	RestoreV2save_6/Const save_6/RestoreV2_39/tensor_names$save_6/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_39Assign%Default/residual_block1/conv2/filterssave_6/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_6/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_40	RestoreV2save_6/Const save_6/RestoreV2_40/tensor_names$save_6/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_6/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_6/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_41	RestoreV2save_6/Const save_6/RestoreV2_41/tensor_names$save_6/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_6/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_6/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_42	RestoreV2save_6/Const save_6/RestoreV2_42/tensor_names$save_6/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_42Assign$Default/residual_block2/conv1/biasessave_6/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_6/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_43	RestoreV2save_6/Const save_6/RestoreV2_43/tensor_names$save_6/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_6/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_6/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_44	RestoreV2save_6/Const save_6/RestoreV2_44/tensor_names$save_6/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_6/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_6/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_45	RestoreV2save_6/Const save_6/RestoreV2_45/tensor_names$save_6/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_45Assign%Default/residual_block2/conv1/filterssave_6/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_6/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_46	RestoreV2save_6/Const save_6/RestoreV2_46/tensor_names$save_6/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_6/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_6/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_47	RestoreV2save_6/Const save_6/RestoreV2_47/tensor_names$save_6/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_6/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_6/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_48	RestoreV2save_6/Const save_6/RestoreV2_48/tensor_names$save_6/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_48Assign$Default/residual_block2/conv2/biasessave_6/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_6/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_49	RestoreV2save_6/Const save_6/RestoreV2_49/tensor_names$save_6/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_6/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_6/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_50	RestoreV2save_6/Const save_6/RestoreV2_50/tensor_names$save_6/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_6/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_6/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_51	RestoreV2save_6/Const save_6/RestoreV2_51/tensor_names$save_6/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_51Assign%Default/residual_block2/conv2/filterssave_6/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_6/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_52	RestoreV2save_6/Const save_6/RestoreV2_52/tensor_names$save_6/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_6/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_6/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_53	RestoreV2save_6/Const save_6/RestoreV2_53/tensor_names$save_6/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_6/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_6/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_54	RestoreV2save_6/Const save_6/RestoreV2_54/tensor_names$save_6/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_54Assign$Default/residual_block3/conv1/biasessave_6/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_6/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_55	RestoreV2save_6/Const save_6/RestoreV2_55/tensor_names$save_6/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_6/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_6/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_56	RestoreV2save_6/Const save_6/RestoreV2_56/tensor_names$save_6/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_6/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_6/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_57	RestoreV2save_6/Const save_6/RestoreV2_57/tensor_names$save_6/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_57Assign%Default/residual_block3/conv1/filterssave_6/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_6/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_58	RestoreV2save_6/Const save_6/RestoreV2_58/tensor_names$save_6/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_6/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_6/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_59	RestoreV2save_6/Const save_6/RestoreV2_59/tensor_names$save_6/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_6/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_6/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_60	RestoreV2save_6/Const save_6/RestoreV2_60/tensor_names$save_6/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_60Assign$Default/residual_block3/conv2/biasessave_6/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_6/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_61	RestoreV2save_6/Const save_6/RestoreV2_61/tensor_names$save_6/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_6/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_6/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_62	RestoreV2save_6/Const save_6/RestoreV2_62/tensor_names$save_6/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_6/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_6/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_63	RestoreV2save_6/Const save_6/RestoreV2_63/tensor_names$save_6/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_63Assign%Default/residual_block3/conv2/filterssave_6/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_6/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_64	RestoreV2save_6/Const save_6/RestoreV2_64/tensor_names$save_6/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_6/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_6/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_65	RestoreV2save_6/Const save_6/RestoreV2_65/tensor_names$save_6/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_6/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_6/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_66	RestoreV2save_6/Const save_6/RestoreV2_66/tensor_names$save_6/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_66Assign$Default/residual_block4/conv1/biasessave_6/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_6/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_67	RestoreV2save_6/Const save_6/RestoreV2_67/tensor_names$save_6/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_6/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_6/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_68	RestoreV2save_6/Const save_6/RestoreV2_68/tensor_names$save_6/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_6/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_6/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_69	RestoreV2save_6/Const save_6/RestoreV2_69/tensor_names$save_6/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_69Assign%Default/residual_block4/conv1/filterssave_6/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_6/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_70	RestoreV2save_6/Const save_6/RestoreV2_70/tensor_names$save_6/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_6/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_6/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_71	RestoreV2save_6/Const save_6/RestoreV2_71/tensor_names$save_6/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_6/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_6/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_72	RestoreV2save_6/Const save_6/RestoreV2_72/tensor_names$save_6/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_6/Assign_72Assign$Default/residual_block4/conv2/biasessave_6/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_6/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_73	RestoreV2save_6/Const save_6/RestoreV2_73/tensor_names$save_6/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_6/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_6/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_6/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_74	RestoreV2save_6/Const save_6/RestoreV2_74/tensor_names$save_6/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_6/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_6/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_6/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_75	RestoreV2save_6/Const save_6/RestoreV2_75/tensor_names$save_6/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_6/Assign_75Assign%Default/residual_block4/conv2/filterssave_6/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_6/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_76	RestoreV2save_6/Const save_6/RestoreV2_76/tensor_names$save_6/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_6/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_6/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_6/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_77	RestoreV2save_6/Const save_6/RestoreV2_77/tensor_names$save_6/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_6/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_6/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_6/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_78	RestoreV2save_6/Const save_6/RestoreV2_78/tensor_names$save_6/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_6/Assign_78Assignbeta1_powersave_6/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_6/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_6/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_6/RestoreV2_79	RestoreV2save_6/Const save_6/RestoreV2_79/tensor_names$save_6/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_6/Assign_79Assignbeta2_powersave_6/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_2^save_6/Assign_3^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_50^save_6/Assign_51^save_6/Assign_52^save_6/Assign_53^save_6/Assign_54^save_6/Assign_55^save_6/Assign_56^save_6/Assign_57^save_6/Assign_58^save_6/Assign_59^save_6/Assign_60^save_6/Assign_61^save_6/Assign_62^save_6/Assign_63^save_6/Assign_64^save_6/Assign_65^save_6/Assign_66^save_6/Assign_67^save_6/Assign_68^save_6/Assign_69^save_6/Assign_70^save_6/Assign_71^save_6/Assign_72^save_6/Assign_73^save_6/Assign_74^save_6/Assign_75^save_6/Assign_76^save_6/Assign_77^save_6/Assign_78^save_6/Assign_79
1
save_6/restore_allNoOp^save_6/restore_shard
R
save_7/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_ccbde78cc4f54940a9978651d18fa47d/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
ē
save_7/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_7/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: 
£
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/control_dependency^save_7/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_7/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_7/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_7/AssignAssign"Default/down_sampling_layer/biasessave_7/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_7/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_1	RestoreV2save_7/Constsave_7/RestoreV2_1/tensor_names#save_7/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_7/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_7/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_7/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_2	RestoreV2save_7/Constsave_7/RestoreV2_2/tensor_names#save_7/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_7/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_7/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_7/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_3	RestoreV2save_7/Constsave_7/RestoreV2_3/tensor_names#save_7/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_7/Assign_3Assign#Default/down_sampling_layer/filterssave_7/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_7/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_4	RestoreV2save_7/Constsave_7/RestoreV2_4/tensor_names#save_7/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_7/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_7/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_7/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_5	RestoreV2save_7/Constsave_7/RestoreV2_5/tensor_names#save_7/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_7/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_7/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_7/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_6	RestoreV2save_7/Constsave_7/RestoreV2_6/tensor_names#save_7/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_7/Assign_6AssignDefault/first_layer/biasessave_7/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_7/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_7	RestoreV2save_7/Constsave_7/RestoreV2_7/tensor_names#save_7/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_7/Assign_7AssignDefault/first_layer/biases/Adamsave_7/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_7/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_8	RestoreV2save_7/Constsave_7/RestoreV2_8/tensor_names#save_7/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_7/Assign_8Assign!Default/first_layer/biases/Adam_1save_7/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_7/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_7/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_7/RestoreV2_9	RestoreV2save_7/Constsave_7/RestoreV2_9/tensor_names#save_7/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_7/Assign_9AssignDefault/first_layer/filterssave_7/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_7/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_10	RestoreV2save_7/Const save_7/RestoreV2_10/tensor_names$save_7/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_7/Assign_10Assign Default/first_layer/filters/Adamsave_7/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_7/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_11	RestoreV2save_7/Const save_7/RestoreV2_11/tensor_names$save_7/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_7/Assign_11Assign"Default/first_layer/filters/Adam_1save_7/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_7/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_12	RestoreV2save_7/Const save_7/RestoreV2_12/tensor_names$save_7/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_7/Assign_12AssignDefault/last_layer/biasessave_7/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_7/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_13	RestoreV2save_7/Const save_7/RestoreV2_13/tensor_names$save_7/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_7/Assign_13AssignDefault/last_layer/biases/Adamsave_7/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_7/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_14	RestoreV2save_7/Const save_7/RestoreV2_14/tensor_names$save_7/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_7/Assign_14Assign Default/last_layer/biases/Adam_1save_7/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_7/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_15	RestoreV2save_7/Const save_7/RestoreV2_15/tensor_names$save_7/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_7/Assign_15AssignDefault/last_layer/filterssave_7/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_7/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_16	RestoreV2save_7/Const save_7/RestoreV2_16/tensor_names$save_7/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_7/Assign_16AssignDefault/last_layer/filters/Adamsave_7/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_7/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_17	RestoreV2save_7/Const save_7/RestoreV2_17/tensor_names$save_7/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_7/Assign_17Assign!Default/last_layer/filters/Adam_1save_7/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_7/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_18	RestoreV2save_7/Const save_7/RestoreV2_18/tensor_names$save_7/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_18Assign$Default/residual_block0/conv1/biasessave_7/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_7/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_19	RestoreV2save_7/Const save_7/RestoreV2_19/tensor_names$save_7/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_7/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_7/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_20	RestoreV2save_7/Const save_7/RestoreV2_20/tensor_names$save_7/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_7/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_7/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_21	RestoreV2save_7/Const save_7/RestoreV2_21/tensor_names$save_7/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_21Assign%Default/residual_block0/conv1/filterssave_7/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_7/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_22	RestoreV2save_7/Const save_7/RestoreV2_22/tensor_names$save_7/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_7/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_7/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_23	RestoreV2save_7/Const save_7/RestoreV2_23/tensor_names$save_7/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_7/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_7/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_24	RestoreV2save_7/Const save_7/RestoreV2_24/tensor_names$save_7/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_24Assign$Default/residual_block0/conv2/biasessave_7/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_7/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_25	RestoreV2save_7/Const save_7/RestoreV2_25/tensor_names$save_7/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_7/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_7/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_26	RestoreV2save_7/Const save_7/RestoreV2_26/tensor_names$save_7/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_7/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_7/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_27	RestoreV2save_7/Const save_7/RestoreV2_27/tensor_names$save_7/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_27Assign%Default/residual_block0/conv2/filterssave_7/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_7/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_28	RestoreV2save_7/Const save_7/RestoreV2_28/tensor_names$save_7/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_7/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_7/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_29	RestoreV2save_7/Const save_7/RestoreV2_29/tensor_names$save_7/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_7/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_7/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_30	RestoreV2save_7/Const save_7/RestoreV2_30/tensor_names$save_7/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_30Assign$Default/residual_block1/conv1/biasessave_7/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_7/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_31	RestoreV2save_7/Const save_7/RestoreV2_31/tensor_names$save_7/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_7/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_7/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_32	RestoreV2save_7/Const save_7/RestoreV2_32/tensor_names$save_7/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_7/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_7/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_33	RestoreV2save_7/Const save_7/RestoreV2_33/tensor_names$save_7/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_33Assign%Default/residual_block1/conv1/filterssave_7/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_7/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_34	RestoreV2save_7/Const save_7/RestoreV2_34/tensor_names$save_7/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_7/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_7/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_35	RestoreV2save_7/Const save_7/RestoreV2_35/tensor_names$save_7/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_7/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_7/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_36	RestoreV2save_7/Const save_7/RestoreV2_36/tensor_names$save_7/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_36Assign$Default/residual_block1/conv2/biasessave_7/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_7/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_37	RestoreV2save_7/Const save_7/RestoreV2_37/tensor_names$save_7/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_7/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_7/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_38	RestoreV2save_7/Const save_7/RestoreV2_38/tensor_names$save_7/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_7/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_7/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_39	RestoreV2save_7/Const save_7/RestoreV2_39/tensor_names$save_7/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_39Assign%Default/residual_block1/conv2/filterssave_7/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_7/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_40	RestoreV2save_7/Const save_7/RestoreV2_40/tensor_names$save_7/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_7/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_7/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_41	RestoreV2save_7/Const save_7/RestoreV2_41/tensor_names$save_7/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_7/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_7/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_42	RestoreV2save_7/Const save_7/RestoreV2_42/tensor_names$save_7/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_42Assign$Default/residual_block2/conv1/biasessave_7/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_7/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_43	RestoreV2save_7/Const save_7/RestoreV2_43/tensor_names$save_7/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_7/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_7/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_44	RestoreV2save_7/Const save_7/RestoreV2_44/tensor_names$save_7/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_7/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_7/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_45	RestoreV2save_7/Const save_7/RestoreV2_45/tensor_names$save_7/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_45Assign%Default/residual_block2/conv1/filterssave_7/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_7/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_46	RestoreV2save_7/Const save_7/RestoreV2_46/tensor_names$save_7/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_7/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_7/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_47	RestoreV2save_7/Const save_7/RestoreV2_47/tensor_names$save_7/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_7/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_7/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_48	RestoreV2save_7/Const save_7/RestoreV2_48/tensor_names$save_7/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_48Assign$Default/residual_block2/conv2/biasessave_7/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_7/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_49	RestoreV2save_7/Const save_7/RestoreV2_49/tensor_names$save_7/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_7/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_7/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_50	RestoreV2save_7/Const save_7/RestoreV2_50/tensor_names$save_7/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_7/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_7/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_51	RestoreV2save_7/Const save_7/RestoreV2_51/tensor_names$save_7/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_51Assign%Default/residual_block2/conv2/filterssave_7/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_7/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_52	RestoreV2save_7/Const save_7/RestoreV2_52/tensor_names$save_7/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_7/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_7/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_53	RestoreV2save_7/Const save_7/RestoreV2_53/tensor_names$save_7/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_7/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_7/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_54	RestoreV2save_7/Const save_7/RestoreV2_54/tensor_names$save_7/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_54Assign$Default/residual_block3/conv1/biasessave_7/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_7/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_55	RestoreV2save_7/Const save_7/RestoreV2_55/tensor_names$save_7/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_7/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_7/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_56	RestoreV2save_7/Const save_7/RestoreV2_56/tensor_names$save_7/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_7/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_7/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_57	RestoreV2save_7/Const save_7/RestoreV2_57/tensor_names$save_7/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_57Assign%Default/residual_block3/conv1/filterssave_7/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_7/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_58	RestoreV2save_7/Const save_7/RestoreV2_58/tensor_names$save_7/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_7/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_7/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_59	RestoreV2save_7/Const save_7/RestoreV2_59/tensor_names$save_7/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_7/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_7/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_60	RestoreV2save_7/Const save_7/RestoreV2_60/tensor_names$save_7/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_60Assign$Default/residual_block3/conv2/biasessave_7/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_7/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_61	RestoreV2save_7/Const save_7/RestoreV2_61/tensor_names$save_7/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_7/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_7/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_62	RestoreV2save_7/Const save_7/RestoreV2_62/tensor_names$save_7/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_7/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_7/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_63	RestoreV2save_7/Const save_7/RestoreV2_63/tensor_names$save_7/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_63Assign%Default/residual_block3/conv2/filterssave_7/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_7/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_64	RestoreV2save_7/Const save_7/RestoreV2_64/tensor_names$save_7/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_7/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_7/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_65	RestoreV2save_7/Const save_7/RestoreV2_65/tensor_names$save_7/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_7/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_7/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_66	RestoreV2save_7/Const save_7/RestoreV2_66/tensor_names$save_7/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_66Assign$Default/residual_block4/conv1/biasessave_7/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_7/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_67	RestoreV2save_7/Const save_7/RestoreV2_67/tensor_names$save_7/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_7/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_7/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_68	RestoreV2save_7/Const save_7/RestoreV2_68/tensor_names$save_7/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_7/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_7/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_69	RestoreV2save_7/Const save_7/RestoreV2_69/tensor_names$save_7/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_69Assign%Default/residual_block4/conv1/filterssave_7/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_7/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_70	RestoreV2save_7/Const save_7/RestoreV2_70/tensor_names$save_7/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_7/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_7/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_71	RestoreV2save_7/Const save_7/RestoreV2_71/tensor_names$save_7/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_7/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_7/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_72	RestoreV2save_7/Const save_7/RestoreV2_72/tensor_names$save_7/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_7/Assign_72Assign$Default/residual_block4/conv2/biasessave_7/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_7/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_73	RestoreV2save_7/Const save_7/RestoreV2_73/tensor_names$save_7/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_7/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_7/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_7/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_74	RestoreV2save_7/Const save_7/RestoreV2_74/tensor_names$save_7/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_7/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_7/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_7/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_75	RestoreV2save_7/Const save_7/RestoreV2_75/tensor_names$save_7/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_7/Assign_75Assign%Default/residual_block4/conv2/filterssave_7/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_7/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_76	RestoreV2save_7/Const save_7/RestoreV2_76/tensor_names$save_7/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_7/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_7/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_7/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_77	RestoreV2save_7/Const save_7/RestoreV2_77/tensor_names$save_7/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_7/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_7/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_7/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_78	RestoreV2save_7/Const save_7/RestoreV2_78/tensor_names$save_7/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_7/Assign_78Assignbeta1_powersave_7/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_7/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_7/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_7/RestoreV2_79	RestoreV2save_7/Const save_7/RestoreV2_79/tensor_names$save_7/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_7/Assign_79Assignbeta2_powersave_7/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_2^save_7/Assign_3^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_50^save_7/Assign_51^save_7/Assign_52^save_7/Assign_53^save_7/Assign_54^save_7/Assign_55^save_7/Assign_56^save_7/Assign_57^save_7/Assign_58^save_7/Assign_59^save_7/Assign_60^save_7/Assign_61^save_7/Assign_62^save_7/Assign_63^save_7/Assign_64^save_7/Assign_65^save_7/Assign_66^save_7/Assign_67^save_7/Assign_68^save_7/Assign_69^save_7/Assign_70^save_7/Assign_71^save_7/Assign_72^save_7/Assign_73^save_7/Assign_74^save_7/Assign_75^save_7/Assign_76^save_7/Assign_77^save_7/Assign_78^save_7/Assign_79
1
save_7/restore_allNoOp^save_7/restore_shard
R
save_8/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_79a7fcc1cdb046ce9f32002926e3d4cd/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
ē
save_8/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_8/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: 
£
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/control_dependency^save_8/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_8/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_8/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_8/AssignAssign"Default/down_sampling_layer/biasessave_8/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_8/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_1	RestoreV2save_8/Constsave_8/RestoreV2_1/tensor_names#save_8/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_8/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_8/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_8/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_2	RestoreV2save_8/Constsave_8/RestoreV2_2/tensor_names#save_8/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_8/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_8/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_8/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_3	RestoreV2save_8/Constsave_8/RestoreV2_3/tensor_names#save_8/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_8/Assign_3Assign#Default/down_sampling_layer/filterssave_8/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_8/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_4	RestoreV2save_8/Constsave_8/RestoreV2_4/tensor_names#save_8/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_8/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_8/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_8/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_5	RestoreV2save_8/Constsave_8/RestoreV2_5/tensor_names#save_8/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_8/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_8/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_8/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_6	RestoreV2save_8/Constsave_8/RestoreV2_6/tensor_names#save_8/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_8/Assign_6AssignDefault/first_layer/biasessave_8/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_8/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_7	RestoreV2save_8/Constsave_8/RestoreV2_7/tensor_names#save_8/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_8/Assign_7AssignDefault/first_layer/biases/Adamsave_8/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_8/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_8	RestoreV2save_8/Constsave_8/RestoreV2_8/tensor_names#save_8/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_8/Assign_8Assign!Default/first_layer/biases/Adam_1save_8/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_8/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_8/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_8/RestoreV2_9	RestoreV2save_8/Constsave_8/RestoreV2_9/tensor_names#save_8/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_8/Assign_9AssignDefault/first_layer/filterssave_8/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_8/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_10	RestoreV2save_8/Const save_8/RestoreV2_10/tensor_names$save_8/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_8/Assign_10Assign Default/first_layer/filters/Adamsave_8/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_8/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_11	RestoreV2save_8/Const save_8/RestoreV2_11/tensor_names$save_8/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_8/Assign_11Assign"Default/first_layer/filters/Adam_1save_8/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_8/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_12	RestoreV2save_8/Const save_8/RestoreV2_12/tensor_names$save_8/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_8/Assign_12AssignDefault/last_layer/biasessave_8/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_8/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_13	RestoreV2save_8/Const save_8/RestoreV2_13/tensor_names$save_8/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_8/Assign_13AssignDefault/last_layer/biases/Adamsave_8/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_8/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_14	RestoreV2save_8/Const save_8/RestoreV2_14/tensor_names$save_8/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_8/Assign_14Assign Default/last_layer/biases/Adam_1save_8/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_8/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_15	RestoreV2save_8/Const save_8/RestoreV2_15/tensor_names$save_8/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_8/Assign_15AssignDefault/last_layer/filterssave_8/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_8/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_16	RestoreV2save_8/Const save_8/RestoreV2_16/tensor_names$save_8/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_8/Assign_16AssignDefault/last_layer/filters/Adamsave_8/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_8/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_17	RestoreV2save_8/Const save_8/RestoreV2_17/tensor_names$save_8/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_8/Assign_17Assign!Default/last_layer/filters/Adam_1save_8/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_8/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_18	RestoreV2save_8/Const save_8/RestoreV2_18/tensor_names$save_8/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_18Assign$Default/residual_block0/conv1/biasessave_8/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_8/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_19	RestoreV2save_8/Const save_8/RestoreV2_19/tensor_names$save_8/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_8/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_8/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_20	RestoreV2save_8/Const save_8/RestoreV2_20/tensor_names$save_8/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_8/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_8/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_21	RestoreV2save_8/Const save_8/RestoreV2_21/tensor_names$save_8/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_21Assign%Default/residual_block0/conv1/filterssave_8/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_8/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_22	RestoreV2save_8/Const save_8/RestoreV2_22/tensor_names$save_8/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_8/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_8/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_23	RestoreV2save_8/Const save_8/RestoreV2_23/tensor_names$save_8/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_8/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_8/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_24	RestoreV2save_8/Const save_8/RestoreV2_24/tensor_names$save_8/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_24Assign$Default/residual_block0/conv2/biasessave_8/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_8/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_25	RestoreV2save_8/Const save_8/RestoreV2_25/tensor_names$save_8/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_8/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_8/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_26	RestoreV2save_8/Const save_8/RestoreV2_26/tensor_names$save_8/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_8/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_8/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_27	RestoreV2save_8/Const save_8/RestoreV2_27/tensor_names$save_8/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_27Assign%Default/residual_block0/conv2/filterssave_8/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_8/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_28	RestoreV2save_8/Const save_8/RestoreV2_28/tensor_names$save_8/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_8/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_8/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_29	RestoreV2save_8/Const save_8/RestoreV2_29/tensor_names$save_8/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_8/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_8/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_30	RestoreV2save_8/Const save_8/RestoreV2_30/tensor_names$save_8/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_30Assign$Default/residual_block1/conv1/biasessave_8/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_8/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_31	RestoreV2save_8/Const save_8/RestoreV2_31/tensor_names$save_8/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_8/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_8/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_32	RestoreV2save_8/Const save_8/RestoreV2_32/tensor_names$save_8/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_8/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_8/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_33	RestoreV2save_8/Const save_8/RestoreV2_33/tensor_names$save_8/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_33Assign%Default/residual_block1/conv1/filterssave_8/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_8/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_34	RestoreV2save_8/Const save_8/RestoreV2_34/tensor_names$save_8/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_8/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_8/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_35	RestoreV2save_8/Const save_8/RestoreV2_35/tensor_names$save_8/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_8/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_8/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_36	RestoreV2save_8/Const save_8/RestoreV2_36/tensor_names$save_8/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_36Assign$Default/residual_block1/conv2/biasessave_8/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_8/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_37	RestoreV2save_8/Const save_8/RestoreV2_37/tensor_names$save_8/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_8/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_8/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_38	RestoreV2save_8/Const save_8/RestoreV2_38/tensor_names$save_8/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_8/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_8/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_39	RestoreV2save_8/Const save_8/RestoreV2_39/tensor_names$save_8/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_39Assign%Default/residual_block1/conv2/filterssave_8/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_8/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_40	RestoreV2save_8/Const save_8/RestoreV2_40/tensor_names$save_8/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_8/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_8/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_41	RestoreV2save_8/Const save_8/RestoreV2_41/tensor_names$save_8/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_8/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_8/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_42	RestoreV2save_8/Const save_8/RestoreV2_42/tensor_names$save_8/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_42Assign$Default/residual_block2/conv1/biasessave_8/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_8/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_43	RestoreV2save_8/Const save_8/RestoreV2_43/tensor_names$save_8/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_8/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_8/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_44	RestoreV2save_8/Const save_8/RestoreV2_44/tensor_names$save_8/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_8/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_8/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_45	RestoreV2save_8/Const save_8/RestoreV2_45/tensor_names$save_8/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_45Assign%Default/residual_block2/conv1/filterssave_8/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_8/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_46	RestoreV2save_8/Const save_8/RestoreV2_46/tensor_names$save_8/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_8/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_8/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_47	RestoreV2save_8/Const save_8/RestoreV2_47/tensor_names$save_8/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_8/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_8/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_48	RestoreV2save_8/Const save_8/RestoreV2_48/tensor_names$save_8/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_48Assign$Default/residual_block2/conv2/biasessave_8/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_8/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_49	RestoreV2save_8/Const save_8/RestoreV2_49/tensor_names$save_8/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_8/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_8/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_50	RestoreV2save_8/Const save_8/RestoreV2_50/tensor_names$save_8/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_8/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_8/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_51	RestoreV2save_8/Const save_8/RestoreV2_51/tensor_names$save_8/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_51Assign%Default/residual_block2/conv2/filterssave_8/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_8/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_52	RestoreV2save_8/Const save_8/RestoreV2_52/tensor_names$save_8/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_8/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_8/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_53	RestoreV2save_8/Const save_8/RestoreV2_53/tensor_names$save_8/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_8/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_8/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_54	RestoreV2save_8/Const save_8/RestoreV2_54/tensor_names$save_8/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_54Assign$Default/residual_block3/conv1/biasessave_8/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_8/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_55	RestoreV2save_8/Const save_8/RestoreV2_55/tensor_names$save_8/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_8/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_8/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_56	RestoreV2save_8/Const save_8/RestoreV2_56/tensor_names$save_8/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_8/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_8/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_57	RestoreV2save_8/Const save_8/RestoreV2_57/tensor_names$save_8/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_57Assign%Default/residual_block3/conv1/filterssave_8/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_8/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_58	RestoreV2save_8/Const save_8/RestoreV2_58/tensor_names$save_8/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_8/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_8/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_59	RestoreV2save_8/Const save_8/RestoreV2_59/tensor_names$save_8/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_8/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_8/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_60	RestoreV2save_8/Const save_8/RestoreV2_60/tensor_names$save_8/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_60Assign$Default/residual_block3/conv2/biasessave_8/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_8/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_61	RestoreV2save_8/Const save_8/RestoreV2_61/tensor_names$save_8/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_8/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_8/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_62	RestoreV2save_8/Const save_8/RestoreV2_62/tensor_names$save_8/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_8/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_8/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_63	RestoreV2save_8/Const save_8/RestoreV2_63/tensor_names$save_8/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_63Assign%Default/residual_block3/conv2/filterssave_8/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_8/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_64	RestoreV2save_8/Const save_8/RestoreV2_64/tensor_names$save_8/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_8/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_8/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_65	RestoreV2save_8/Const save_8/RestoreV2_65/tensor_names$save_8/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_8/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_8/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_66	RestoreV2save_8/Const save_8/RestoreV2_66/tensor_names$save_8/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_66Assign$Default/residual_block4/conv1/biasessave_8/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_8/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_67	RestoreV2save_8/Const save_8/RestoreV2_67/tensor_names$save_8/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_8/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_8/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_68	RestoreV2save_8/Const save_8/RestoreV2_68/tensor_names$save_8/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_8/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_8/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_69	RestoreV2save_8/Const save_8/RestoreV2_69/tensor_names$save_8/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_69Assign%Default/residual_block4/conv1/filterssave_8/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_8/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_70	RestoreV2save_8/Const save_8/RestoreV2_70/tensor_names$save_8/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_8/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_8/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_71	RestoreV2save_8/Const save_8/RestoreV2_71/tensor_names$save_8/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_8/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_8/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_72	RestoreV2save_8/Const save_8/RestoreV2_72/tensor_names$save_8/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_8/Assign_72Assign$Default/residual_block4/conv2/biasessave_8/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_8/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_73	RestoreV2save_8/Const save_8/RestoreV2_73/tensor_names$save_8/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_8/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_8/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_8/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_74	RestoreV2save_8/Const save_8/RestoreV2_74/tensor_names$save_8/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_8/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_8/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_8/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_75	RestoreV2save_8/Const save_8/RestoreV2_75/tensor_names$save_8/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_8/Assign_75Assign%Default/residual_block4/conv2/filterssave_8/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_8/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_76	RestoreV2save_8/Const save_8/RestoreV2_76/tensor_names$save_8/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_8/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_8/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_8/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_77	RestoreV2save_8/Const save_8/RestoreV2_77/tensor_names$save_8/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_8/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_8/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_8/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_78	RestoreV2save_8/Const save_8/RestoreV2_78/tensor_names$save_8/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_8/Assign_78Assignbeta1_powersave_8/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_8/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_8/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_8/RestoreV2_79	RestoreV2save_8/Const save_8/RestoreV2_79/tensor_names$save_8/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_8/Assign_79Assignbeta2_powersave_8/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_2^save_8/Assign_3^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_50^save_8/Assign_51^save_8/Assign_52^save_8/Assign_53^save_8/Assign_54^save_8/Assign_55^save_8/Assign_56^save_8/Assign_57^save_8/Assign_58^save_8/Assign_59^save_8/Assign_60^save_8/Assign_61^save_8/Assign_62^save_8/Assign_63^save_8/Assign_64^save_8/Assign_65^save_8/Assign_66^save_8/Assign_67^save_8/Assign_68^save_8/Assign_69^save_8/Assign_70^save_8/Assign_71^save_8/Assign_72^save_8/Assign_73^save_8/Assign_74^save_8/Assign_75^save_8/Assign_76^save_8/Assign_77^save_8/Assign_78^save_8/Assign_79
1
save_8/restore_allNoOp^save_8/restore_shard
R
save_9/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_21c96c952166416f9da65c7c741e307c/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
ē
save_9/SaveV2/tensor_namesConst*
valueBPB"Default/down_sampling_layer/biasesB'Default/down_sampling_layer/biases/AdamB)Default/down_sampling_layer/biases/Adam_1B#Default/down_sampling_layer/filtersB(Default/down_sampling_layer/filters/AdamB*Default/down_sampling_layer/filters/Adam_1BDefault/first_layer/biasesBDefault/first_layer/biases/AdamB!Default/first_layer/biases/Adam_1BDefault/first_layer/filtersB Default/first_layer/filters/AdamB"Default/first_layer/filters/Adam_1BDefault/last_layer/biasesBDefault/last_layer/biases/AdamB Default/last_layer/biases/Adam_1BDefault/last_layer/filtersBDefault/last_layer/filters/AdamB!Default/last_layer/filters/Adam_1B$Default/residual_block0/conv1/biasesB)Default/residual_block0/conv1/biases/AdamB+Default/residual_block0/conv1/biases/Adam_1B%Default/residual_block0/conv1/filtersB*Default/residual_block0/conv1/filters/AdamB,Default/residual_block0/conv1/filters/Adam_1B$Default/residual_block0/conv2/biasesB)Default/residual_block0/conv2/biases/AdamB+Default/residual_block0/conv2/biases/Adam_1B%Default/residual_block0/conv2/filtersB*Default/residual_block0/conv2/filters/AdamB,Default/residual_block0/conv2/filters/Adam_1B$Default/residual_block1/conv1/biasesB)Default/residual_block1/conv1/biases/AdamB+Default/residual_block1/conv1/biases/Adam_1B%Default/residual_block1/conv1/filtersB*Default/residual_block1/conv1/filters/AdamB,Default/residual_block1/conv1/filters/Adam_1B$Default/residual_block1/conv2/biasesB)Default/residual_block1/conv2/biases/AdamB+Default/residual_block1/conv2/biases/Adam_1B%Default/residual_block1/conv2/filtersB*Default/residual_block1/conv2/filters/AdamB,Default/residual_block1/conv2/filters/Adam_1B$Default/residual_block2/conv1/biasesB)Default/residual_block2/conv1/biases/AdamB+Default/residual_block2/conv1/biases/Adam_1B%Default/residual_block2/conv1/filtersB*Default/residual_block2/conv1/filters/AdamB,Default/residual_block2/conv1/filters/Adam_1B$Default/residual_block2/conv2/biasesB)Default/residual_block2/conv2/biases/AdamB+Default/residual_block2/conv2/biases/Adam_1B%Default/residual_block2/conv2/filtersB*Default/residual_block2/conv2/filters/AdamB,Default/residual_block2/conv2/filters/Adam_1B$Default/residual_block3/conv1/biasesB)Default/residual_block3/conv1/biases/AdamB+Default/residual_block3/conv1/biases/Adam_1B%Default/residual_block3/conv1/filtersB*Default/residual_block3/conv1/filters/AdamB,Default/residual_block3/conv1/filters/Adam_1B$Default/residual_block3/conv2/biasesB)Default/residual_block3/conv2/biases/AdamB+Default/residual_block3/conv2/biases/Adam_1B%Default/residual_block3/conv2/filtersB*Default/residual_block3/conv2/filters/AdamB,Default/residual_block3/conv2/filters/Adam_1B$Default/residual_block4/conv1/biasesB)Default/residual_block4/conv1/biases/AdamB+Default/residual_block4/conv1/biases/Adam_1B%Default/residual_block4/conv1/filtersB*Default/residual_block4/conv1/filters/AdamB,Default/residual_block4/conv1/filters/Adam_1B$Default/residual_block4/conv2/biasesB)Default/residual_block4/conv2/biases/AdamB+Default/residual_block4/conv2/biases/Adam_1B%Default/residual_block4/conv2/filtersB*Default/residual_block4/conv2/filters/AdamB,Default/residual_block4/conv2/filters/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:P

save_9/SaveV2/shape_and_slicesConst*µ
value«BØPB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:P
Ī
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slices"Default/down_sampling_layer/biases'Default/down_sampling_layer/biases/Adam)Default/down_sampling_layer/biases/Adam_1#Default/down_sampling_layer/filters(Default/down_sampling_layer/filters/Adam*Default/down_sampling_layer/filters/Adam_1Default/first_layer/biasesDefault/first_layer/biases/Adam!Default/first_layer/biases/Adam_1Default/first_layer/filters Default/first_layer/filters/Adam"Default/first_layer/filters/Adam_1Default/last_layer/biasesDefault/last_layer/biases/Adam Default/last_layer/biases/Adam_1Default/last_layer/filtersDefault/last_layer/filters/Adam!Default/last_layer/filters/Adam_1$Default/residual_block0/conv1/biases)Default/residual_block0/conv1/biases/Adam+Default/residual_block0/conv1/biases/Adam_1%Default/residual_block0/conv1/filters*Default/residual_block0/conv1/filters/Adam,Default/residual_block0/conv1/filters/Adam_1$Default/residual_block0/conv2/biases)Default/residual_block0/conv2/biases/Adam+Default/residual_block0/conv2/biases/Adam_1%Default/residual_block0/conv2/filters*Default/residual_block0/conv2/filters/Adam,Default/residual_block0/conv2/filters/Adam_1$Default/residual_block1/conv1/biases)Default/residual_block1/conv1/biases/Adam+Default/residual_block1/conv1/biases/Adam_1%Default/residual_block1/conv1/filters*Default/residual_block1/conv1/filters/Adam,Default/residual_block1/conv1/filters/Adam_1$Default/residual_block1/conv2/biases)Default/residual_block1/conv2/biases/Adam+Default/residual_block1/conv2/biases/Adam_1%Default/residual_block1/conv2/filters*Default/residual_block1/conv2/filters/Adam,Default/residual_block1/conv2/filters/Adam_1$Default/residual_block2/conv1/biases)Default/residual_block2/conv1/biases/Adam+Default/residual_block2/conv1/biases/Adam_1%Default/residual_block2/conv1/filters*Default/residual_block2/conv1/filters/Adam,Default/residual_block2/conv1/filters/Adam_1$Default/residual_block2/conv2/biases)Default/residual_block2/conv2/biases/Adam+Default/residual_block2/conv2/biases/Adam_1%Default/residual_block2/conv2/filters*Default/residual_block2/conv2/filters/Adam,Default/residual_block2/conv2/filters/Adam_1$Default/residual_block3/conv1/biases)Default/residual_block3/conv1/biases/Adam+Default/residual_block3/conv1/biases/Adam_1%Default/residual_block3/conv1/filters*Default/residual_block3/conv1/filters/Adam,Default/residual_block3/conv1/filters/Adam_1$Default/residual_block3/conv2/biases)Default/residual_block3/conv2/biases/Adam+Default/residual_block3/conv2/biases/Adam_1%Default/residual_block3/conv2/filters*Default/residual_block3/conv2/filters/Adam,Default/residual_block3/conv2/filters/Adam_1$Default/residual_block4/conv1/biases)Default/residual_block4/conv1/biases/Adam+Default/residual_block4/conv1/biases/Adam_1%Default/residual_block4/conv1/filters*Default/residual_block4/conv1/filters/Adam,Default/residual_block4/conv1/filters/Adam_1$Default/residual_block4/conv2/biases)Default/residual_block4/conv2/biases/Adam+Default/residual_block4/conv2/biases/Adam_1%Default/residual_block4/conv2/filters*Default/residual_block4/conv2/filters/Adam,Default/residual_block4/conv2/filters/Adam_1beta1_powerbeta2_power*^
dtypesT
R2P

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: 
£
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
N*
T0*

axis *
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(

save_9/IdentityIdentitysave_9/Const^save_9/control_dependency^save_9/MergeV2Checkpoints*
T0*
_output_shapes
: 

save_9/RestoreV2/tensor_namesConst*7
value.B,B"Default/down_sampling_layer/biases*
dtype0*
_output_shapes
:
j
!save_9/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ņ
save_9/AssignAssign"Default/down_sampling_layer/biasessave_9/RestoreV2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_9/RestoreV2_1/tensor_namesConst*<
value3B1B'Default/down_sampling_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_1	RestoreV2save_9/Constsave_9/RestoreV2_1/tensor_names#save_9/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_9/Assign_1Assign'Default/down_sampling_layer/biases/Adamsave_9/RestoreV2_1*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_9/RestoreV2_2/tensor_namesConst*>
value5B3B)Default/down_sampling_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_2	RestoreV2save_9/Constsave_9/RestoreV2_2/tensor_names#save_9/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_9/Assign_2Assign)Default/down_sampling_layer/biases/Adam_1save_9/RestoreV2_2*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@Default/down_sampling_layer/biases*
_output_shapes
:

save_9/RestoreV2_3/tensor_namesConst*8
value/B-B#Default/down_sampling_layer/filters*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_3	RestoreV2save_9/Constsave_9/RestoreV2_3/tensor_names#save_9/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ä
save_9/Assign_3Assign#Default/down_sampling_layer/filterssave_9/RestoreV2_3*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_9/RestoreV2_4/tensor_namesConst*=
value4B2B(Default/down_sampling_layer/filters/Adam*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_4	RestoreV2save_9/Constsave_9/RestoreV2_4/tensor_names#save_9/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
é
save_9/Assign_4Assign(Default/down_sampling_layer/filters/Adamsave_9/RestoreV2_4*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_9/RestoreV2_5/tensor_namesConst*?
value6B4B*Default/down_sampling_layer/filters/Adam_1*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_5	RestoreV2save_9/Constsave_9/RestoreV2_5/tensor_names#save_9/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ė
save_9/Assign_5Assign*Default/down_sampling_layer/filters/Adam_1save_9/RestoreV2_5*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@Default/down_sampling_layer/filters*&
_output_shapes
:

save_9/RestoreV2_6/tensor_namesConst*/
value&B$BDefault/first_layer/biases*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_6	RestoreV2save_9/Constsave_9/RestoreV2_6/tensor_names#save_9/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_9/Assign_6AssignDefault/first_layer/biasessave_9/RestoreV2_6*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_9/RestoreV2_7/tensor_namesConst*4
value+B)BDefault/first_layer/biases/Adam*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_7	RestoreV2save_9/Constsave_9/RestoreV2_7/tensor_names#save_9/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_9/Assign_7AssignDefault/first_layer/biases/Adamsave_9/RestoreV2_7*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_9/RestoreV2_8/tensor_namesConst*6
value-B+B!Default/first_layer/biases/Adam_1*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_8	RestoreV2save_9/Constsave_9/RestoreV2_8/tensor_names#save_9/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_9/Assign_8Assign!Default/first_layer/biases/Adam_1save_9/RestoreV2_8*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/first_layer/biases*
_output_shapes
: 

save_9/RestoreV2_9/tensor_namesConst*0
value'B%BDefault/first_layer/filters*
dtype0*
_output_shapes
:
l
#save_9/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:

save_9/RestoreV2_9	RestoreV2save_9/Constsave_9/RestoreV2_9/tensor_names#save_9/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_9/Assign_9AssignDefault/first_layer/filterssave_9/RestoreV2_9*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_9/RestoreV2_10/tensor_namesConst*5
value,B*B Default/first_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_10	RestoreV2save_9/Const save_9/RestoreV2_10/tensor_names$save_9/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_9/Assign_10Assign Default/first_layer/filters/Adamsave_9/RestoreV2_10*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_9/RestoreV2_11/tensor_namesConst*7
value.B,B"Default/first_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_11	RestoreV2save_9/Const save_9/RestoreV2_11/tensor_names$save_9/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
Ż
save_9/Assign_11Assign"Default/first_layer/filters/Adam_1save_9/RestoreV2_11*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*&
_output_shapes
: 

 save_9/RestoreV2_12/tensor_namesConst*.
value%B#BDefault/last_layer/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_12	RestoreV2save_9/Const save_9/RestoreV2_12/tensor_names$save_9/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ę
save_9/Assign_12AssignDefault/last_layer/biasessave_9/RestoreV2_12*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_9/RestoreV2_13/tensor_namesConst*3
value*B(BDefault/last_layer/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_13	RestoreV2save_9/Const save_9/RestoreV2_13/tensor_names$save_9/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
Ė
save_9/Assign_13AssignDefault/last_layer/biases/Adamsave_9/RestoreV2_13*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_9/RestoreV2_14/tensor_namesConst*5
value,B*B Default/last_layer/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_14	RestoreV2save_9/Const save_9/RestoreV2_14/tensor_names$save_9/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ķ
save_9/Assign_14Assign Default/last_layer/biases/Adam_1save_9/RestoreV2_14*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@Default/last_layer/biases*
_output_shapes
:K

 save_9/RestoreV2_15/tensor_namesConst*/
value&B$BDefault/last_layer/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_15	RestoreV2save_9/Const save_9/RestoreV2_15/tensor_names$save_9/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
Ō
save_9/Assign_15AssignDefault/last_layer/filterssave_9/RestoreV2_15*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_9/RestoreV2_16/tensor_namesConst*4
value+B)BDefault/last_layer/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_16	RestoreV2save_9/Const save_9/RestoreV2_16/tensor_names$save_9/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ł
save_9/Assign_16AssignDefault/last_layer/filters/Adamsave_9/RestoreV2_16*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_9/RestoreV2_17/tensor_namesConst*6
value-B+B!Default/last_layer/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_17	RestoreV2save_9/Const save_9/RestoreV2_17/tensor_names$save_9/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ū
save_9/Assign_17Assign!Default/last_layer/filters/Adam_1save_9/RestoreV2_17*
T0*
validate_shape(*
use_locking(*-
_class#
!loc:@Default/last_layer/filters*&
_output_shapes
:>K

 save_9/RestoreV2_18/tensor_namesConst*9
value0B.B$Default/residual_block0/conv1/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_18	RestoreV2save_9/Const save_9/RestoreV2_18/tensor_names$save_9/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_18Assign$Default/residual_block0/conv1/biasessave_9/RestoreV2_18*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_9/RestoreV2_19/tensor_namesConst*>
value5B3B)Default/residual_block0/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_19	RestoreV2save_9/Const save_9/RestoreV2_19/tensor_names$save_9/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_19Assign)Default/residual_block0/conv1/biases/Adamsave_9/RestoreV2_19*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_9/RestoreV2_20/tensor_namesConst*@
value7B5B+Default/residual_block0/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_20	RestoreV2save_9/Const save_9/RestoreV2_20/tensor_names$save_9/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_20Assign+Default/residual_block0/conv1/biases/Adam_1save_9/RestoreV2_20*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv1/biases*
_output_shapes
:"

 save_9/RestoreV2_21/tensor_namesConst*:
value1B/B%Default/residual_block0/conv1/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_21	RestoreV2save_9/Const save_9/RestoreV2_21/tensor_names$save_9/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_21Assign%Default/residual_block0/conv1/filterssave_9/RestoreV2_21*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_9/RestoreV2_22/tensor_namesConst*?
value6B4B*Default/residual_block0/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_22	RestoreV2save_9/Const save_9/RestoreV2_22/tensor_names$save_9/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_22Assign*Default/residual_block0/conv1/filters/Adamsave_9/RestoreV2_22*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_9/RestoreV2_23/tensor_namesConst*A
value8B6B,Default/residual_block0/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_23	RestoreV2save_9/Const save_9/RestoreV2_23/tensor_names$save_9/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_23Assign,Default/residual_block0/conv1/filters/Adam_1save_9/RestoreV2_23*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv1/filters*&
_output_shapes
: "

 save_9/RestoreV2_24/tensor_namesConst*9
value0B.B$Default/residual_block0/conv2/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_24	RestoreV2save_9/Const save_9/RestoreV2_24/tensor_names$save_9/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_24Assign$Default/residual_block0/conv2/biasessave_9/RestoreV2_24*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_9/RestoreV2_25/tensor_namesConst*>
value5B3B)Default/residual_block0/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_25	RestoreV2save_9/Const save_9/RestoreV2_25/tensor_names$save_9/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_25Assign)Default/residual_block0/conv2/biases/Adamsave_9/RestoreV2_25*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_9/RestoreV2_26/tensor_namesConst*@
value7B5B+Default/residual_block0/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_26	RestoreV2save_9/Const save_9/RestoreV2_26/tensor_names$save_9/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_26Assign+Default/residual_block0/conv2/biases/Adam_1save_9/RestoreV2_26*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block0/conv2/biases*
_output_shapes
:"

 save_9/RestoreV2_27/tensor_namesConst*:
value1B/B%Default/residual_block0/conv2/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_27	RestoreV2save_9/Const save_9/RestoreV2_27/tensor_names$save_9/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_27Assign%Default/residual_block0/conv2/filterssave_9/RestoreV2_27*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_9/RestoreV2_28/tensor_namesConst*?
value6B4B*Default/residual_block0/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_28	RestoreV2save_9/Const save_9/RestoreV2_28/tensor_names$save_9/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_28Assign*Default/residual_block0/conv2/filters/Adamsave_9/RestoreV2_28*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_9/RestoreV2_29/tensor_namesConst*A
value8B6B,Default/residual_block0/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_29	RestoreV2save_9/Const save_9/RestoreV2_29/tensor_names$save_9/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_29Assign,Default/residual_block0/conv2/filters/Adam_1save_9/RestoreV2_29*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block0/conv2/filters*&
_output_shapes
:""

 save_9/RestoreV2_30/tensor_namesConst*9
value0B.B$Default/residual_block1/conv1/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_30	RestoreV2save_9/Const save_9/RestoreV2_30/tensor_names$save_9/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_30Assign$Default/residual_block1/conv1/biasessave_9/RestoreV2_30*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_9/RestoreV2_31/tensor_namesConst*>
value5B3B)Default/residual_block1/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_31	RestoreV2save_9/Const save_9/RestoreV2_31/tensor_names$save_9/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_31Assign)Default/residual_block1/conv1/biases/Adamsave_9/RestoreV2_31*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_9/RestoreV2_32/tensor_namesConst*@
value7B5B+Default/residual_block1/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_32/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_32	RestoreV2save_9/Const save_9/RestoreV2_32/tensor_names$save_9/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_32Assign+Default/residual_block1/conv1/biases/Adam_1save_9/RestoreV2_32*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv1/biases*
_output_shapes
:&

 save_9/RestoreV2_33/tensor_namesConst*:
value1B/B%Default/residual_block1/conv1/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_33	RestoreV2save_9/Const save_9/RestoreV2_33/tensor_names$save_9/RestoreV2_33/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_33Assign%Default/residual_block1/conv1/filterssave_9/RestoreV2_33*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_9/RestoreV2_34/tensor_namesConst*?
value6B4B*Default/residual_block1/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_34	RestoreV2save_9/Const save_9/RestoreV2_34/tensor_names$save_9/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_34Assign*Default/residual_block1/conv1/filters/Adamsave_9/RestoreV2_34*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_9/RestoreV2_35/tensor_namesConst*A
value8B6B,Default/residual_block1/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_35	RestoreV2save_9/Const save_9/RestoreV2_35/tensor_names$save_9/RestoreV2_35/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_35Assign,Default/residual_block1/conv1/filters/Adam_1save_9/RestoreV2_35*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv1/filters*&
_output_shapes
:"&

 save_9/RestoreV2_36/tensor_namesConst*9
value0B.B$Default/residual_block1/conv2/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_36	RestoreV2save_9/Const save_9/RestoreV2_36/tensor_names$save_9/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_36Assign$Default/residual_block1/conv2/biasessave_9/RestoreV2_36*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_9/RestoreV2_37/tensor_namesConst*>
value5B3B)Default/residual_block1/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_37	RestoreV2save_9/Const save_9/RestoreV2_37/tensor_names$save_9/RestoreV2_37/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_37Assign)Default/residual_block1/conv2/biases/Adamsave_9/RestoreV2_37*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_9/RestoreV2_38/tensor_namesConst*@
value7B5B+Default/residual_block1/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_38	RestoreV2save_9/Const save_9/RestoreV2_38/tensor_names$save_9/RestoreV2_38/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_38Assign+Default/residual_block1/conv2/biases/Adam_1save_9/RestoreV2_38*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block1/conv2/biases*
_output_shapes
:&

 save_9/RestoreV2_39/tensor_namesConst*:
value1B/B%Default/residual_block1/conv2/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_39/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_39	RestoreV2save_9/Const save_9/RestoreV2_39/tensor_names$save_9/RestoreV2_39/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_39Assign%Default/residual_block1/conv2/filterssave_9/RestoreV2_39*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_9/RestoreV2_40/tensor_namesConst*?
value6B4B*Default/residual_block1/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_40	RestoreV2save_9/Const save_9/RestoreV2_40/tensor_names$save_9/RestoreV2_40/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_40Assign*Default/residual_block1/conv2/filters/Adamsave_9/RestoreV2_40*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_9/RestoreV2_41/tensor_namesConst*A
value8B6B,Default/residual_block1/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_41	RestoreV2save_9/Const save_9/RestoreV2_41/tensor_names$save_9/RestoreV2_41/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_41Assign,Default/residual_block1/conv2/filters/Adam_1save_9/RestoreV2_41*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block1/conv2/filters*&
_output_shapes
:&&

 save_9/RestoreV2_42/tensor_namesConst*9
value0B.B$Default/residual_block2/conv1/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_42	RestoreV2save_9/Const save_9/RestoreV2_42/tensor_names$save_9/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_42Assign$Default/residual_block2/conv1/biasessave_9/RestoreV2_42*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_9/RestoreV2_43/tensor_namesConst*>
value5B3B)Default/residual_block2/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_43	RestoreV2save_9/Const save_9/RestoreV2_43/tensor_names$save_9/RestoreV2_43/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_43Assign)Default/residual_block2/conv1/biases/Adamsave_9/RestoreV2_43*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_9/RestoreV2_44/tensor_namesConst*@
value7B5B+Default/residual_block2/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_44/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_44	RestoreV2save_9/Const save_9/RestoreV2_44/tensor_names$save_9/RestoreV2_44/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_44Assign+Default/residual_block2/conv1/biases/Adam_1save_9/RestoreV2_44*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv1/biases*
_output_shapes
:,

 save_9/RestoreV2_45/tensor_namesConst*:
value1B/B%Default/residual_block2/conv1/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_45/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_45	RestoreV2save_9/Const save_9/RestoreV2_45/tensor_names$save_9/RestoreV2_45/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_45Assign%Default/residual_block2/conv1/filterssave_9/RestoreV2_45*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_9/RestoreV2_46/tensor_namesConst*?
value6B4B*Default/residual_block2/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_46/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_46	RestoreV2save_9/Const save_9/RestoreV2_46/tensor_names$save_9/RestoreV2_46/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_46Assign*Default/residual_block2/conv1/filters/Adamsave_9/RestoreV2_46*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_9/RestoreV2_47/tensor_namesConst*A
value8B6B,Default/residual_block2/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_47/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_47	RestoreV2save_9/Const save_9/RestoreV2_47/tensor_names$save_9/RestoreV2_47/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_47Assign,Default/residual_block2/conv1/filters/Adam_1save_9/RestoreV2_47*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv1/filters*&
_output_shapes
:&,

 save_9/RestoreV2_48/tensor_namesConst*9
value0B.B$Default/residual_block2/conv2/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_48/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_48	RestoreV2save_9/Const save_9/RestoreV2_48/tensor_names$save_9/RestoreV2_48/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_48Assign$Default/residual_block2/conv2/biasessave_9/RestoreV2_48*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_9/RestoreV2_49/tensor_namesConst*>
value5B3B)Default/residual_block2/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_49/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_49	RestoreV2save_9/Const save_9/RestoreV2_49/tensor_names$save_9/RestoreV2_49/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_49Assign)Default/residual_block2/conv2/biases/Adamsave_9/RestoreV2_49*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_9/RestoreV2_50/tensor_namesConst*@
value7B5B+Default/residual_block2/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_50/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_50	RestoreV2save_9/Const save_9/RestoreV2_50/tensor_names$save_9/RestoreV2_50/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_50Assign+Default/residual_block2/conv2/biases/Adam_1save_9/RestoreV2_50*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block2/conv2/biases*
_output_shapes
:,

 save_9/RestoreV2_51/tensor_namesConst*:
value1B/B%Default/residual_block2/conv2/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_51/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_51	RestoreV2save_9/Const save_9/RestoreV2_51/tensor_names$save_9/RestoreV2_51/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_51Assign%Default/residual_block2/conv2/filterssave_9/RestoreV2_51*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_9/RestoreV2_52/tensor_namesConst*?
value6B4B*Default/residual_block2/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_52/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_52	RestoreV2save_9/Const save_9/RestoreV2_52/tensor_names$save_9/RestoreV2_52/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_52Assign*Default/residual_block2/conv2/filters/Adamsave_9/RestoreV2_52*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_9/RestoreV2_53/tensor_namesConst*A
value8B6B,Default/residual_block2/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_53/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_53	RestoreV2save_9/Const save_9/RestoreV2_53/tensor_names$save_9/RestoreV2_53/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_53Assign,Default/residual_block2/conv2/filters/Adam_1save_9/RestoreV2_53*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block2/conv2/filters*&
_output_shapes
:,,

 save_9/RestoreV2_54/tensor_namesConst*9
value0B.B$Default/residual_block3/conv1/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_54/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_54	RestoreV2save_9/Const save_9/RestoreV2_54/tensor_names$save_9/RestoreV2_54/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_54Assign$Default/residual_block3/conv1/biasessave_9/RestoreV2_54*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_9/RestoreV2_55/tensor_namesConst*>
value5B3B)Default/residual_block3/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_55/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_55	RestoreV2save_9/Const save_9/RestoreV2_55/tensor_names$save_9/RestoreV2_55/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_55Assign)Default/residual_block3/conv1/biases/Adamsave_9/RestoreV2_55*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_9/RestoreV2_56/tensor_namesConst*@
value7B5B+Default/residual_block3/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_56/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_56	RestoreV2save_9/Const save_9/RestoreV2_56/tensor_names$save_9/RestoreV2_56/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_56Assign+Default/residual_block3/conv1/biases/Adam_1save_9/RestoreV2_56*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv1/biases*
_output_shapes
:4

 save_9/RestoreV2_57/tensor_namesConst*:
value1B/B%Default/residual_block3/conv1/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_57/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_57	RestoreV2save_9/Const save_9/RestoreV2_57/tensor_names$save_9/RestoreV2_57/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_57Assign%Default/residual_block3/conv1/filterssave_9/RestoreV2_57*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_9/RestoreV2_58/tensor_namesConst*?
value6B4B*Default/residual_block3/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_58/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_58	RestoreV2save_9/Const save_9/RestoreV2_58/tensor_names$save_9/RestoreV2_58/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_58Assign*Default/residual_block3/conv1/filters/Adamsave_9/RestoreV2_58*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_9/RestoreV2_59/tensor_namesConst*A
value8B6B,Default/residual_block3/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_59/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_59	RestoreV2save_9/Const save_9/RestoreV2_59/tensor_names$save_9/RestoreV2_59/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_59Assign,Default/residual_block3/conv1/filters/Adam_1save_9/RestoreV2_59*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv1/filters*&
_output_shapes
:,4

 save_9/RestoreV2_60/tensor_namesConst*9
value0B.B$Default/residual_block3/conv2/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_60/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_60	RestoreV2save_9/Const save_9/RestoreV2_60/tensor_names$save_9/RestoreV2_60/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_60Assign$Default/residual_block3/conv2/biasessave_9/RestoreV2_60*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_9/RestoreV2_61/tensor_namesConst*>
value5B3B)Default/residual_block3/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_61/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_61	RestoreV2save_9/Const save_9/RestoreV2_61/tensor_names$save_9/RestoreV2_61/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_61Assign)Default/residual_block3/conv2/biases/Adamsave_9/RestoreV2_61*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_9/RestoreV2_62/tensor_namesConst*@
value7B5B+Default/residual_block3/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_62/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_62	RestoreV2save_9/Const save_9/RestoreV2_62/tensor_names$save_9/RestoreV2_62/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_62Assign+Default/residual_block3/conv2/biases/Adam_1save_9/RestoreV2_62*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block3/conv2/biases*
_output_shapes
:4

 save_9/RestoreV2_63/tensor_namesConst*:
value1B/B%Default/residual_block3/conv2/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_63/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_63	RestoreV2save_9/Const save_9/RestoreV2_63/tensor_names$save_9/RestoreV2_63/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_63Assign%Default/residual_block3/conv2/filterssave_9/RestoreV2_63*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_9/RestoreV2_64/tensor_namesConst*?
value6B4B*Default/residual_block3/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_64/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_64	RestoreV2save_9/Const save_9/RestoreV2_64/tensor_names$save_9/RestoreV2_64/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_64Assign*Default/residual_block3/conv2/filters/Adamsave_9/RestoreV2_64*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_9/RestoreV2_65/tensor_namesConst*A
value8B6B,Default/residual_block3/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_65/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_65	RestoreV2save_9/Const save_9/RestoreV2_65/tensor_names$save_9/RestoreV2_65/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_65Assign,Default/residual_block3/conv2/filters/Adam_1save_9/RestoreV2_65*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block3/conv2/filters*&
_output_shapes
:44

 save_9/RestoreV2_66/tensor_namesConst*9
value0B.B$Default/residual_block4/conv1/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_66/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_66	RestoreV2save_9/Const save_9/RestoreV2_66/tensor_names$save_9/RestoreV2_66/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_66Assign$Default/residual_block4/conv1/biasessave_9/RestoreV2_66*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_9/RestoreV2_67/tensor_namesConst*>
value5B3B)Default/residual_block4/conv1/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_67/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_67	RestoreV2save_9/Const save_9/RestoreV2_67/tensor_names$save_9/RestoreV2_67/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_67Assign)Default/residual_block4/conv1/biases/Adamsave_9/RestoreV2_67*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_9/RestoreV2_68/tensor_namesConst*@
value7B5B+Default/residual_block4/conv1/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_68/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_68	RestoreV2save_9/Const save_9/RestoreV2_68/tensor_names$save_9/RestoreV2_68/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_68Assign+Default/residual_block4/conv1/biases/Adam_1save_9/RestoreV2_68*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv1/biases*
_output_shapes
:>

 save_9/RestoreV2_69/tensor_namesConst*:
value1B/B%Default/residual_block4/conv1/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_69/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_69	RestoreV2save_9/Const save_9/RestoreV2_69/tensor_names$save_9/RestoreV2_69/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_69Assign%Default/residual_block4/conv1/filterssave_9/RestoreV2_69*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_9/RestoreV2_70/tensor_namesConst*?
value6B4B*Default/residual_block4/conv1/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_70/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_70	RestoreV2save_9/Const save_9/RestoreV2_70/tensor_names$save_9/RestoreV2_70/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_70Assign*Default/residual_block4/conv1/filters/Adamsave_9/RestoreV2_70*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_9/RestoreV2_71/tensor_namesConst*A
value8B6B,Default/residual_block4/conv1/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_71/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_71	RestoreV2save_9/Const save_9/RestoreV2_71/tensor_names$save_9/RestoreV2_71/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_71Assign,Default/residual_block4/conv1/filters/Adam_1save_9/RestoreV2_71*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv1/filters*&
_output_shapes
:4>

 save_9/RestoreV2_72/tensor_namesConst*9
value0B.B$Default/residual_block4/conv2/biases*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_72/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_72	RestoreV2save_9/Const save_9/RestoreV2_72/tensor_names$save_9/RestoreV2_72/shape_and_slices*
dtypes
2*
_output_shapes
:
Ü
save_9/Assign_72Assign$Default/residual_block4/conv2/biasessave_9/RestoreV2_72*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_9/RestoreV2_73/tensor_namesConst*>
value5B3B)Default/residual_block4/conv2/biases/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_73/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_73	RestoreV2save_9/Const save_9/RestoreV2_73/tensor_names$save_9/RestoreV2_73/shape_and_slices*
dtypes
2*
_output_shapes
:
į
save_9/Assign_73Assign)Default/residual_block4/conv2/biases/Adamsave_9/RestoreV2_73*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_9/RestoreV2_74/tensor_namesConst*@
value7B5B+Default/residual_block4/conv2/biases/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_74/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_74	RestoreV2save_9/Const save_9/RestoreV2_74/tensor_names$save_9/RestoreV2_74/shape_and_slices*
dtypes
2*
_output_shapes
:
ć
save_9/Assign_74Assign+Default/residual_block4/conv2/biases/Adam_1save_9/RestoreV2_74*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@Default/residual_block4/conv2/biases*
_output_shapes
:>

 save_9/RestoreV2_75/tensor_namesConst*:
value1B/B%Default/residual_block4/conv2/filters*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_75/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_75	RestoreV2save_9/Const save_9/RestoreV2_75/tensor_names$save_9/RestoreV2_75/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_9/Assign_75Assign%Default/residual_block4/conv2/filterssave_9/RestoreV2_75*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_9/RestoreV2_76/tensor_namesConst*?
value6B4B*Default/residual_block4/conv2/filters/Adam*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_76/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_76	RestoreV2save_9/Const save_9/RestoreV2_76/tensor_names$save_9/RestoreV2_76/shape_and_slices*
dtypes
2*
_output_shapes
:
ļ
save_9/Assign_76Assign*Default/residual_block4/conv2/filters/Adamsave_9/RestoreV2_76*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>

 save_9/RestoreV2_77/tensor_namesConst*A
value8B6B,Default/residual_block4/conv2/filters/Adam_1*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_77/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_77	RestoreV2save_9/Const save_9/RestoreV2_77/tensor_names$save_9/RestoreV2_77/shape_and_slices*
dtypes
2*
_output_shapes
:
ń
save_9/Assign_77Assign,Default/residual_block4/conv2/filters/Adam_1save_9/RestoreV2_77*
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@Default/residual_block4/conv2/filters*&
_output_shapes
:>>
t
 save_9/RestoreV2_78/tensor_namesConst* 
valueBBbeta1_power*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_78/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_78	RestoreV2save_9/Const save_9/RestoreV2_78/tensor_names$save_9/RestoreV2_78/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_9/Assign_78Assignbeta1_powersave_9/RestoreV2_78*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 
t
 save_9/RestoreV2_79/tensor_namesConst* 
valueBBbeta2_power*
dtype0*
_output_shapes
:
m
$save_9/RestoreV2_79/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
”
save_9/RestoreV2_79	RestoreV2save_9/Const save_9/RestoreV2_79/tensor_names$save_9/RestoreV2_79/shape_and_slices*
dtypes
2*
_output_shapes
:
¶
save_9/Assign_79Assignbeta2_powersave_9/RestoreV2_79*
T0*
validate_shape(*
use_locking(*.
_class$
" loc:@Default/first_layer/filters*
_output_shapes
: 

save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_2^save_9/Assign_3^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_50^save_9/Assign_51^save_9/Assign_52^save_9/Assign_53^save_9/Assign_54^save_9/Assign_55^save_9/Assign_56^save_9/Assign_57^save_9/Assign_58^save_9/Assign_59^save_9/Assign_60^save_9/Assign_61^save_9/Assign_62^save_9/Assign_63^save_9/Assign_64^save_9/Assign_65^save_9/Assign_66^save_9/Assign_67^save_9/Assign_68^save_9/Assign_69^save_9/Assign_70^save_9/Assign_71^save_9/Assign_72^save_9/Assign_73^save_9/Assign_74^save_9/Assign_75^save_9/Assign_76^save_9/Assign_77^save_9/Assign_78^save_9/Assign_79
1
save_9/restore_allNoOp^save_9/restore_shard"B
save_9/Const:0save_9/Identity:0save_9/restore_all (5 @F8"śV
	variablesģVéV
g
Default/first_layer/filters:0"Default/first_layer/filters/Assign"Default/first_layer/filters/read:0
d
Default/first_layer/biases:0!Default/first_layer/biases/Assign!Default/first_layer/biases/read:0

'Default/residual_block0/conv1/filters:0,Default/residual_block0/conv1/filters/Assign,Default/residual_block0/conv1/filters/read:0

&Default/residual_block0/conv1/biases:0+Default/residual_block0/conv1/biases/Assign+Default/residual_block0/conv1/biases/read:0

'Default/residual_block0/conv2/filters:0,Default/residual_block0/conv2/filters/Assign,Default/residual_block0/conv2/filters/read:0

&Default/residual_block0/conv2/biases:0+Default/residual_block0/conv2/biases/Assign+Default/residual_block0/conv2/biases/read:0

'Default/residual_block1/conv1/filters:0,Default/residual_block1/conv1/filters/Assign,Default/residual_block1/conv1/filters/read:0

&Default/residual_block1/conv1/biases:0+Default/residual_block1/conv1/biases/Assign+Default/residual_block1/conv1/biases/read:0

'Default/residual_block1/conv2/filters:0,Default/residual_block1/conv2/filters/Assign,Default/residual_block1/conv2/filters/read:0

&Default/residual_block1/conv2/biases:0+Default/residual_block1/conv2/biases/Assign+Default/residual_block1/conv2/biases/read:0

'Default/residual_block2/conv1/filters:0,Default/residual_block2/conv1/filters/Assign,Default/residual_block2/conv1/filters/read:0

&Default/residual_block2/conv1/biases:0+Default/residual_block2/conv1/biases/Assign+Default/residual_block2/conv1/biases/read:0

'Default/residual_block2/conv2/filters:0,Default/residual_block2/conv2/filters/Assign,Default/residual_block2/conv2/filters/read:0

&Default/residual_block2/conv2/biases:0+Default/residual_block2/conv2/biases/Assign+Default/residual_block2/conv2/biases/read:0

'Default/residual_block3/conv1/filters:0,Default/residual_block3/conv1/filters/Assign,Default/residual_block3/conv1/filters/read:0

&Default/residual_block3/conv1/biases:0+Default/residual_block3/conv1/biases/Assign+Default/residual_block3/conv1/biases/read:0

'Default/residual_block3/conv2/filters:0,Default/residual_block3/conv2/filters/Assign,Default/residual_block3/conv2/filters/read:0

&Default/residual_block3/conv2/biases:0+Default/residual_block3/conv2/biases/Assign+Default/residual_block3/conv2/biases/read:0

'Default/residual_block4/conv1/filters:0,Default/residual_block4/conv1/filters/Assign,Default/residual_block4/conv1/filters/read:0

&Default/residual_block4/conv1/biases:0+Default/residual_block4/conv1/biases/Assign+Default/residual_block4/conv1/biases/read:0

'Default/residual_block4/conv2/filters:0,Default/residual_block4/conv2/filters/Assign,Default/residual_block4/conv2/filters/read:0

&Default/residual_block4/conv2/biases:0+Default/residual_block4/conv2/biases/Assign+Default/residual_block4/conv2/biases/read:0
d
Default/last_layer/filters:0!Default/last_layer/filters/Assign!Default/last_layer/filters/read:0
a
Default/last_layer/biases:0 Default/last_layer/biases/Assign Default/last_layer/biases/read:0

%Default/down_sampling_layer/filters:0*Default/down_sampling_layer/filters/Assign*Default/down_sampling_layer/filters/read:0
|
$Default/down_sampling_layer/biases:0)Default/down_sampling_layer/biases/Assign)Default/down_sampling_layer/biases/read:0
7
beta1_power:0beta1_power/Assignbeta1_power/read:0
7
beta2_power:0beta2_power/Assignbeta2_power/read:0
v
"Default/first_layer/filters/Adam:0'Default/first_layer/filters/Adam/Assign'Default/first_layer/filters/Adam/read:0
|
$Default/first_layer/filters/Adam_1:0)Default/first_layer/filters/Adam_1/Assign)Default/first_layer/filters/Adam_1/read:0
s
!Default/first_layer/biases/Adam:0&Default/first_layer/biases/Adam/Assign&Default/first_layer/biases/Adam/read:0
y
#Default/first_layer/biases/Adam_1:0(Default/first_layer/biases/Adam_1/Assign(Default/first_layer/biases/Adam_1/read:0

,Default/residual_block0/conv1/filters/Adam:01Default/residual_block0/conv1/filters/Adam/Assign1Default/residual_block0/conv1/filters/Adam/read:0

.Default/residual_block0/conv1/filters/Adam_1:03Default/residual_block0/conv1/filters/Adam_1/Assign3Default/residual_block0/conv1/filters/Adam_1/read:0

+Default/residual_block0/conv1/biases/Adam:00Default/residual_block0/conv1/biases/Adam/Assign0Default/residual_block0/conv1/biases/Adam/read:0

-Default/residual_block0/conv1/biases/Adam_1:02Default/residual_block0/conv1/biases/Adam_1/Assign2Default/residual_block0/conv1/biases/Adam_1/read:0

,Default/residual_block0/conv2/filters/Adam:01Default/residual_block0/conv2/filters/Adam/Assign1Default/residual_block0/conv2/filters/Adam/read:0

.Default/residual_block0/conv2/filters/Adam_1:03Default/residual_block0/conv2/filters/Adam_1/Assign3Default/residual_block0/conv2/filters/Adam_1/read:0

+Default/residual_block0/conv2/biases/Adam:00Default/residual_block0/conv2/biases/Adam/Assign0Default/residual_block0/conv2/biases/Adam/read:0

-Default/residual_block0/conv2/biases/Adam_1:02Default/residual_block0/conv2/biases/Adam_1/Assign2Default/residual_block0/conv2/biases/Adam_1/read:0

,Default/residual_block1/conv1/filters/Adam:01Default/residual_block1/conv1/filters/Adam/Assign1Default/residual_block1/conv1/filters/Adam/read:0

.Default/residual_block1/conv1/filters/Adam_1:03Default/residual_block1/conv1/filters/Adam_1/Assign3Default/residual_block1/conv1/filters/Adam_1/read:0

+Default/residual_block1/conv1/biases/Adam:00Default/residual_block1/conv1/biases/Adam/Assign0Default/residual_block1/conv1/biases/Adam/read:0

-Default/residual_block1/conv1/biases/Adam_1:02Default/residual_block1/conv1/biases/Adam_1/Assign2Default/residual_block1/conv1/biases/Adam_1/read:0

,Default/residual_block1/conv2/filters/Adam:01Default/residual_block1/conv2/filters/Adam/Assign1Default/residual_block1/conv2/filters/Adam/read:0

.Default/residual_block1/conv2/filters/Adam_1:03Default/residual_block1/conv2/filters/Adam_1/Assign3Default/residual_block1/conv2/filters/Adam_1/read:0

+Default/residual_block1/conv2/biases/Adam:00Default/residual_block1/conv2/biases/Adam/Assign0Default/residual_block1/conv2/biases/Adam/read:0

-Default/residual_block1/conv2/biases/Adam_1:02Default/residual_block1/conv2/biases/Adam_1/Assign2Default/residual_block1/conv2/biases/Adam_1/read:0

,Default/residual_block2/conv1/filters/Adam:01Default/residual_block2/conv1/filters/Adam/Assign1Default/residual_block2/conv1/filters/Adam/read:0

.Default/residual_block2/conv1/filters/Adam_1:03Default/residual_block2/conv1/filters/Adam_1/Assign3Default/residual_block2/conv1/filters/Adam_1/read:0

+Default/residual_block2/conv1/biases/Adam:00Default/residual_block2/conv1/biases/Adam/Assign0Default/residual_block2/conv1/biases/Adam/read:0

-Default/residual_block2/conv1/biases/Adam_1:02Default/residual_block2/conv1/biases/Adam_1/Assign2Default/residual_block2/conv1/biases/Adam_1/read:0

,Default/residual_block2/conv2/filters/Adam:01Default/residual_block2/conv2/filters/Adam/Assign1Default/residual_block2/conv2/filters/Adam/read:0

.Default/residual_block2/conv2/filters/Adam_1:03Default/residual_block2/conv2/filters/Adam_1/Assign3Default/residual_block2/conv2/filters/Adam_1/read:0

+Default/residual_block2/conv2/biases/Adam:00Default/residual_block2/conv2/biases/Adam/Assign0Default/residual_block2/conv2/biases/Adam/read:0

-Default/residual_block2/conv2/biases/Adam_1:02Default/residual_block2/conv2/biases/Adam_1/Assign2Default/residual_block2/conv2/biases/Adam_1/read:0

,Default/residual_block3/conv1/filters/Adam:01Default/residual_block3/conv1/filters/Adam/Assign1Default/residual_block3/conv1/filters/Adam/read:0

.Default/residual_block3/conv1/filters/Adam_1:03Default/residual_block3/conv1/filters/Adam_1/Assign3Default/residual_block3/conv1/filters/Adam_1/read:0

+Default/residual_block3/conv1/biases/Adam:00Default/residual_block3/conv1/biases/Adam/Assign0Default/residual_block3/conv1/biases/Adam/read:0

-Default/residual_block3/conv1/biases/Adam_1:02Default/residual_block3/conv1/biases/Adam_1/Assign2Default/residual_block3/conv1/biases/Adam_1/read:0

,Default/residual_block3/conv2/filters/Adam:01Default/residual_block3/conv2/filters/Adam/Assign1Default/residual_block3/conv2/filters/Adam/read:0

.Default/residual_block3/conv2/filters/Adam_1:03Default/residual_block3/conv2/filters/Adam_1/Assign3Default/residual_block3/conv2/filters/Adam_1/read:0

+Default/residual_block3/conv2/biases/Adam:00Default/residual_block3/conv2/biases/Adam/Assign0Default/residual_block3/conv2/biases/Adam/read:0

-Default/residual_block3/conv2/biases/Adam_1:02Default/residual_block3/conv2/biases/Adam_1/Assign2Default/residual_block3/conv2/biases/Adam_1/read:0

,Default/residual_block4/conv1/filters/Adam:01Default/residual_block4/conv1/filters/Adam/Assign1Default/residual_block4/conv1/filters/Adam/read:0

.Default/residual_block4/conv1/filters/Adam_1:03Default/residual_block4/conv1/filters/Adam_1/Assign3Default/residual_block4/conv1/filters/Adam_1/read:0

+Default/residual_block4/conv1/biases/Adam:00Default/residual_block4/conv1/biases/Adam/Assign0Default/residual_block4/conv1/biases/Adam/read:0

-Default/residual_block4/conv1/biases/Adam_1:02Default/residual_block4/conv1/biases/Adam_1/Assign2Default/residual_block4/conv1/biases/Adam_1/read:0

,Default/residual_block4/conv2/filters/Adam:01Default/residual_block4/conv2/filters/Adam/Assign1Default/residual_block4/conv2/filters/Adam/read:0

.Default/residual_block4/conv2/filters/Adam_1:03Default/residual_block4/conv2/filters/Adam_1/Assign3Default/residual_block4/conv2/filters/Adam_1/read:0

+Default/residual_block4/conv2/biases/Adam:00Default/residual_block4/conv2/biases/Adam/Assign0Default/residual_block4/conv2/biases/Adam/read:0

-Default/residual_block4/conv2/biases/Adam_1:02Default/residual_block4/conv2/biases/Adam_1/Assign2Default/residual_block4/conv2/biases/Adam_1/read:0
s
!Default/last_layer/filters/Adam:0&Default/last_layer/filters/Adam/Assign&Default/last_layer/filters/Adam/read:0
y
#Default/last_layer/filters/Adam_1:0(Default/last_layer/filters/Adam_1/Assign(Default/last_layer/filters/Adam_1/read:0
p
 Default/last_layer/biases/Adam:0%Default/last_layer/biases/Adam/Assign%Default/last_layer/biases/Adam/read:0
v
"Default/last_layer/biases/Adam_1:0'Default/last_layer/biases/Adam_1/Assign'Default/last_layer/biases/Adam_1/read:0

*Default/down_sampling_layer/filters/Adam:0/Default/down_sampling_layer/filters/Adam/Assign/Default/down_sampling_layer/filters/Adam/read:0

,Default/down_sampling_layer/filters/Adam_1:01Default/down_sampling_layer/filters/Adam_1/Assign1Default/down_sampling_layer/filters/Adam_1/read:0

)Default/down_sampling_layer/biases/Adam:0.Default/down_sampling_layer/biases/Adam/Assign.Default/down_sampling_layer/biases/Adam/read:0

+Default/down_sampling_layer/biases/Adam_1:00Default/down_sampling_layer/biases/Adam_1/Assign0Default/down_sampling_layer/biases/Adam_1/read:0"“
trainable_variables
g
Default/first_layer/filters:0"Default/first_layer/filters/Assign"Default/first_layer/filters/read:0
d
Default/first_layer/biases:0!Default/first_layer/biases/Assign!Default/first_layer/biases/read:0

'Default/residual_block0/conv1/filters:0,Default/residual_block0/conv1/filters/Assign,Default/residual_block0/conv1/filters/read:0

&Default/residual_block0/conv1/biases:0+Default/residual_block0/conv1/biases/Assign+Default/residual_block0/conv1/biases/read:0

'Default/residual_block0/conv2/filters:0,Default/residual_block0/conv2/filters/Assign,Default/residual_block0/conv2/filters/read:0

&Default/residual_block0/conv2/biases:0+Default/residual_block0/conv2/biases/Assign+Default/residual_block0/conv2/biases/read:0

'Default/residual_block1/conv1/filters:0,Default/residual_block1/conv1/filters/Assign,Default/residual_block1/conv1/filters/read:0

&Default/residual_block1/conv1/biases:0+Default/residual_block1/conv1/biases/Assign+Default/residual_block1/conv1/biases/read:0

'Default/residual_block1/conv2/filters:0,Default/residual_block1/conv2/filters/Assign,Default/residual_block1/conv2/filters/read:0

&Default/residual_block1/conv2/biases:0+Default/residual_block1/conv2/biases/Assign+Default/residual_block1/conv2/biases/read:0

'Default/residual_block2/conv1/filters:0,Default/residual_block2/conv1/filters/Assign,Default/residual_block2/conv1/filters/read:0

&Default/residual_block2/conv1/biases:0+Default/residual_block2/conv1/biases/Assign+Default/residual_block2/conv1/biases/read:0

'Default/residual_block2/conv2/filters:0,Default/residual_block2/conv2/filters/Assign,Default/residual_block2/conv2/filters/read:0

&Default/residual_block2/conv2/biases:0+Default/residual_block2/conv2/biases/Assign+Default/residual_block2/conv2/biases/read:0

'Default/residual_block3/conv1/filters:0,Default/residual_block3/conv1/filters/Assign,Default/residual_block3/conv1/filters/read:0

&Default/residual_block3/conv1/biases:0+Default/residual_block3/conv1/biases/Assign+Default/residual_block3/conv1/biases/read:0

'Default/residual_block3/conv2/filters:0,Default/residual_block3/conv2/filters/Assign,Default/residual_block3/conv2/filters/read:0

&Default/residual_block3/conv2/biases:0+Default/residual_block3/conv2/biases/Assign+Default/residual_block3/conv2/biases/read:0

'Default/residual_block4/conv1/filters:0,Default/residual_block4/conv1/filters/Assign,Default/residual_block4/conv1/filters/read:0

&Default/residual_block4/conv1/biases:0+Default/residual_block4/conv1/biases/Assign+Default/residual_block4/conv1/biases/read:0

'Default/residual_block4/conv2/filters:0,Default/residual_block4/conv2/filters/Assign,Default/residual_block4/conv2/filters/read:0

&Default/residual_block4/conv2/biases:0+Default/residual_block4/conv2/biases/Assign+Default/residual_block4/conv2/biases/read:0
d
Default/last_layer/filters:0!Default/last_layer/filters/Assign!Default/last_layer/filters/read:0
a
Default/last_layer/biases:0 Default/last_layer/biases/Assign Default/last_layer/biases/read:0

%Default/down_sampling_layer/filters:0*Default/down_sampling_layer/filters/Assign*Default/down_sampling_layer/filters/read:0
|
$Default/down_sampling_layer/biases:0)Default/down_sampling_layer/biases/Assign)Default/down_sampling_layer/biases/read:0"
train_op

Adam*
bicubic_imagesļ
P
input_images@
Placeholder_3:0+’’’’’’’’’’’’’’’’’’’’’’’’’’’
"
shape
Placeholder_5:0[
output_bicubic_imagesB
ResizeBicubic_1:0+’’’’’’’’’’’’’’’’’’’’’’’’’’’tensorflow/serving/predict*Ī
serving_defaultŗ
P
input_images@
Placeholder_3:0+’’’’’’’’’’’’’’’’’’’’’’’’’’’J
output_images9
add_35:0+’’’’’’’’’’’’’’’’’’’’’’’’’’’tensorflow/serving/predict