#pragma once

#define XNN_WRAP_IN_LAMBDA(body) [&] {body}
#define IF_ELSE(cond, if_block, else_block) IfElse(XNN_WRAP_IN_LAMBDA(cond;),XNN_WRAP_IN_LAMBDA(if_block), XNN_WRAP_IN_LAMBDA(else_block) );
#define IF(cond, if_block) If(XNN_WRAP_IN_LAMBDA(cond;),XNN_WRAP_IN_LAMBDA(if_block));
#define DO_WHILE(body, cond) DoWhile(XNN_WRAP_IN_LAMBDA(body), XNN_WRAP_IN_LAMBDA(cond;));
#define WHILE(cond, body) While(XNN_WRAP_IN_LAMBDA(cond;), XNN_WRAP_IN_LAMBDA(body));

