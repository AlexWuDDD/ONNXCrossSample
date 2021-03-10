#include<stdio.h>

int check_sys()
{
    int num = 1;
    if (*(char *)&num == 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int main()
{
    if (check_sys() == 1)
    {
        printf("您的机器是小端字节序\n");
    }
    else
    {
        printf("您的机器是大端字节序\n");
    }
    return 0;
}