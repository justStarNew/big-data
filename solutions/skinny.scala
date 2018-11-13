object Palindromics extends App {
    
    def R(n:Int): Int = n.toString.reverse.toInt
    def palindromic(n:Int): Boolean = (n == R(n))
    
    val start = args(0).toInt
    val stop = args(1).toInt

    for ( i <- start to stop ) {
        if (R(i*i) == R(i)*R(i) && !palindromic(i)) {
            println(i)
        }
    }
}
Palindromics.main(Array("100","1000"))
