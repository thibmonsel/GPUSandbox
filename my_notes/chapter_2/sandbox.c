#include <stdio.h>
#include <stdlib.h>

int main()
{
    int a = 30;
    int *ptra = &a;

    printf("%p", &a);
    printf("%d", *ptra);

    return 0;
}