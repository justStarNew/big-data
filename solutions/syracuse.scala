def syracuse1 ( n:Int ) {
    var i = n
    while ( i > 1 ) {
        if ( i % 2 == 0 ) {
            i = i / 2
        } else {
            i = 3*i+1
        }
        println(i)
    }
}
syracuse1(33)


def syracuse ( n:Int ) {
    println(n)
    if ( n == 1 )
      1
    else {
       if ( n % 2 == 0 ) {
            syracuse( n / 2)
        } else {
            syracuse( 3*n+1)
        }
    }
}
syracuse2(33)
