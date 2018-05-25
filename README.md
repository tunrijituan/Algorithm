## Algorithm
### 算法在编程代码过程中有着举足轻重的作用，如何使得代码更高效，简洁是程序员一直追求的方向。本篇以剑指offer为例，对其中涉及的部分编程问题提出了自己的解决方案。
#### 1.二维数组的查找
##### 在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```
  public boolean Find(int target, int [][] array) {
          int col = array.length-1;
	        int i = 0;
	        while((col >= 0)&& (i < array[0].length)){
	            if(array[col][i] > target){
	                col--;
	            }else if(array[col][i] < target){
	                i++;
	            }else{
	                return true;
	            }
	        }
	        return false;
    }
```
* 由于二维数组是有序的，从左到右递增，从上到下递增。由此可考虑从左下角开始查找，如果target的值>array[col][i],行加1.如果target<array[col][i]，列减1;
#### 2.重建二叉树
##### 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
```
  /**
 *Definition for binary tree
 *public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
import java.util.HashMap;
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if(in.length==0||pre.length==0)
	        	return null;
	   	    HashMap< Integer, Integer> map=new HashMap<Integer, Integer>();
	   	    for(int i=0;i<in.length;i++){
	   	    	map.put(in[i], i);
	   	    }
	   	     return preIn(pre,0,pre.length-1,in,0,in.length-1,map);
	    }
	 public static TreeNode preIn(int[] pre,int pi,int pj,int[] n,int ni,int nj,HashMap<Integer,Integer> map){
		 
	        if(pi>pj){
	            return null;
	        }
	        TreeNode head=new TreeNode(pre[pi]);
	        int index=map.get(pre[pi]);
	        head.left=preIn(pre,pi+1,pi+index-ni,n,ni,index-1,map);
	        head.right=preIn(pre,pi+index-ni+1,pj,n,index+1,nj,map);
	        return head;
	    }
}
```
* 我们知道由中序遍历和前序遍历或者中序遍历和后续遍历都可以唯一确定一个二叉树


