# Algorithm
算法在编程代码过程中有着举足轻重的作用，如何使得代码更高效，简洁是程序员一直追求的方向。本篇以剑指offer为例，对其中涉及的部分编程问题提出了自己的解决方案。
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
* 我们知道由中序遍历和前序遍历或者中序遍历和后续遍历都可以唯一确定一个二叉树,前序遍历为DLR,中序遍历LDR,后序遍历LRD.由此可以确定前序遍历的第一个值肯定是根节点，再由此数值对应中序位置，左边的为左子树，右边的为右子树。依次遍历得到最后的二叉树。
#### 3.两个栈实现队列
##### 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
```
public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
   	 public  void push(int node) {
		stack1.push(node);
    }
	public  int pop() {
		  while(!stack1.isEmpty()){
	            stack2.push(stack1.pop());
	        }
	        int node=stack2.pop();
	        while(!stack2.isEmpty()){
	            stack1.push(stack2.pop());
	        }
	        return node;
	    }
}
```
* 首先确定队列和栈的特性，队列先进先出，栈先进后出。由此进队的时候直接进入栈stack1中即可，出队时，则先将stack1中的全部pop到stack2中，取出stack2的第一个值即为出队值，出队后要将stack2再依次取回到stack1，这样做是为了保证进队出队顺序不变.（日常操作时，有时候会遗忘进队1，2，出队1，进队3）
#### 4.斐波那契数列（青蛙跳台阶是一类问题）
##### 现在要求输入一个整数n，请你输出斐波那契数列的第n项。
```
public int Fibonacci(int n) {
                 int a=1;int b=1;int c=0;
		if(n==0){
			return 0;
		}
		if(n==1||n==2)
			return 1;
		for(int i=3;i<n+1;i++){
			c=a+b;
			a=b;
			b=c;
		}
		return c;
    }
```
```   
	if(n==0){
		return 0;
		}
		if(n==1||n==2)
		return 1;
	return Fibonacci(n-1)+Fibonacci(n-2);   
```
* 可以使用递归和非递归两种方法解决这个问题
#### 5.变态跳台阶
##### 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
```
	Math.pow(2,n-1);
```
* 不论跳几步，最后一个台阶是一定要跳的，其他台阶跳不跳两种情况，所以所有的跳法为2^(n-1);
#### 6.合并两个排序的链表
##### 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
```
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
         if(list1==null){
	            return list2;
	        }
		 else if(list2==null){
	            return list1;
	        }
	        ListNode newHead = null;
	        if(list1.val<list2.val){
	        	newHead = list1;
	        	newHead.next = Merge(list1.next,list2);
	        }else{
	        	newHead = list2;
	        	newHead.next = Merge(list1,list2.next);
	        }
	        return newHead;
    }
}
```
* 递归法：如果list1.val<list2.val,新节点为list1，然后将list1.next与list2继续进行比较直到完成所有比较。
```
if(list1 == null){
            return list2;
        }
        if(list2 == null){
            return list1;
        }
        ListNode mergeHead = null;
        ListNode current = null;     
        while(list1!=null && list2!=null){
            if(list1.val <= list2.val){
                if(mergeHead == null){
                   mergeHead = current = list1;
                }else{
                   current.next = list1;
                   current = current.next;
                }
                list1 = list1.next;
            }else{
                if(mergeHead == null){
                   mergeHead = current = list2;
                }else{
                   current.next = list2;
                   current = current.next;
                }
                list2 = list2.next;
            }
        }
        if(list1 == null){
            current.next = list2;
        }else{
            current.next = list1;
        }
        return mergeHead;
```
* 非递归解法：使用while循环不断比较
#### 7.树的子结构
##### 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）。
```
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
      if(root2==null) return false;
	        if(root1==null && root2!=null) return false;      
	        boolean flag = false;
	        if(root1.val==root2.val){
	            flag = isSubTree(root1,root2);
	        }
	        if(!flag){
	            flag = HasSubtree(root1.left, root2);
	            if(!flag){
	                flag = HasSubtree(root1.right, root2);
	            }
	        }
	        return flag;
	    }
	     /**
	      *判断是否完全相同  
	      *@param root1
	      *@param root2
	      *@return
	      */
	    private boolean isSubTree(TreeNode root1, TreeNode root2) {
	        if(root2==null) return true;
	        if(root1==null && root2!=null) return false;      
	        if(root1.val==root2.val){
	            return isSubTree(root1.left, root2.left) && isSubTree(root1.right, root2.right);
	        }else{
	        return false;
	        }
	    }
}
```
* 是不是树的子结构，也就是找到root1的某部分是否与root2的结构完全相同，使用递归解决，分别比较根节点，左子树，右子树。
#### 6.最小的K个数
##### 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。
```
public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> list=new ArrayList<Integer>();
        	if(input.length<k){
			return list;
		}
		Map<Integer, Integer> map=new TreeMap<Integer, Integer>();
		for(int i=0;i<input.length;i++){
			map.put(i, input[i]);
		}
		 List<Entry<Integer, Integer>> list1 = new ArrayList<Entry<Integer, Integer>>(map.entrySet());
		  Collections.sort(list1,new Comparator<Map.Entry<Integer,Integer>>() {  
	            //升序排序  
	            public int compare(Entry<Integer, Integer> o1, Entry<Integer, Integer> o2) {  
	                return o1.getValue().compareTo(o2.getValue());  
	            }  
	        });  
	         int count=0;
	        if(list1.size()<k){
	        	 for (Entry<Integer, Integer> e: list1) {  
	 	        		list.add(e.getValue());
	 	        	}
	         }
	         else{
	        for (Entry<Integer, Integer> e: list1) {  
	        	if(count<k){
	        		list.add(e.getValue());
	 	           count++;
	        	}
	        	else
	        		break;
	        }  
	         }
		 return list;	        
	 }
}
```
* 对于这个问题，最直接的想法就是排序，取前K个，该解决方法使用Treemap进行升序操作，然后截取了前K个即为最小数
#### 7.Top K问题
##### 从20亿个数据中取出最大（或者最小）的20个数
```
  public int findKthLargest(int[] nums, int k) {
  PriorityQueue<Integer> minQueue = new PriorityQueue<>(k);
  for (int num : nums) {
    if (minQueue.size() < k || num > minQueue.peek())
      minQueue.offer(num);
    if (minQueue.size() > k)
      minQueue.poll();
  }
  return minQueue.peek();
}
```
* 解决这么大数据的取值问题，如果采用排序方法简直得不偿失，20亿个数据得排序到什么时候，因此直接的方法就是采用基于比较的方法进行处理。先利用小顶堆维护当前扫描到的最大100个数，其后每一次的扫描到的元素，若大于堆顶，则入堆，然后删除堆顶；依此往复，直至扫描完所有元素。

#### 8.连续子数组的最大和
##### 数组中的数有正有负，取得连续子数组的最大和
```
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
       if(array.length==0)
		return 0;
		int sum=array[0];
		int next=array[0];
		for(int i=1;i<array.length;i++){
		next = (next < 0) ? array[i] : next + array[i];
	        sum = (next > sum) ? next : sum;
		}
		return sum; 
    }
}
```
* 如果全是正数则非常容易，现在数组中有可能存在负数的情况，因此在循环过程中，要先求第一个局部的最大和，然后与下一个局部最大和进行比较；举例说明6,3,-2,-7,15,-1,-2,-2; sum和next都先取第一个数值，接下来next=9，sum=9;next=7,sum=9;next=0,sum=9;next=15,sum=15;
#### 9.数组中出现次数超过一半的数字
##### 例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
```
public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
         if(array.length==1)
        	return array[0];
        Arrays.sort(array);
        int count=1; 
        for(int i=0;i<array.length-1;i++){
        	if(array[i]==array[i+1]){
        		count++;
        		if(count>array.length/2)
            		return array[i];
        	}
        	else{
        		count=1;
        	}
        }
        return 0;
    }
}
```
* 如果只有一个数则直接返回，如果length>=2,则先排序，然后for循环进行比较；
#### 10.数组中的逆序对
##### 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007。
```
		public static int InversePairs(int [] array) {
		 int count=0;
		 List<Integer> list=new ArrayList<>();
		 for(int i=0;i<array.length;i++){
			 list.add(array[i]);
		 	}
		while(!list.isEmpty()){
			int a=min(list);
			count=count+a;
			list.remove(list.get(a));
			}
		 return count%1000000007;      
		}
	 	public static int min(List<Integer> array){
		 int Min=0;
		 for(int i=1;i<array.size();i++){
			 if(array.get(Min)>array.get(i)){
				 Min=i;
			 }
		 	}
		 return Min;
	 	}
```
* 数据量不大的情况下可以使用暴力法，两个for循环进行比较
```
//数组中的逆序对（分治思想，归并排序思想解决）
	    public static int InversePairsVery(int[] array){
	        if(array==null||array.length<=1)
	            return 0;
	        int[] copy = new int[array.length];
	        for(int i=0;i<array.length;i++){
	            copy[i] = array[i];
	        }
	        return mergeCount(array, copy, 0, array.length-1);
	    }
	    
	    public static int mergeCount(int[] array, int[] copy, int start, int end){
	        if(start==end){
	            copy[start] = array[start];
	            return 0;
	        }
	        int mid = (start+end)>>1;
	        int leftCount = mergeCount(copy, array, start, mid);
	        int rightCount = mergeCount(copy, array, mid+1, end);	        
	        int i = mid;//i初始化为前半段最后一个数字的下标
	        int j = end;//j初始化为后半段最后一个数字的下标
	        int index = end;//辅助数组复制的数组的最后一个数字的下标
	        int count = 0;//计数--逆序对的数目
	        while(i>=start&&j>=mid+1){
	            if(array[i]>array[j]){
	                copy[index--] = array[i--];
	                count += j-mid;
	            }else{
	                copy[index--] = array[j--];
	            }
	        }
	        for(;i>=start;i--){
	            copy[index--] = array[i];
	        }
	        for(;j>=mid+1;j--){
	            copy[index--] = array[j];
	        }
	        return leftCount+rightCount+count;
	    }
```
* 分治处理，分成两部分，分别求左右数组中逆序对的个数，然后求左右数组整合后新数组逆序对个数；逆序对的总数=左边数组中的逆序对的数量+右边数组中逆序对的数量+左右结合成新的顺序数组时中出现的逆序对的数量；举例来说更清楚些：对数组{1,2,3,4,5,6,7,0}求逆序对，
* （1）1,2,3,4|5,6,7,0| 两部分
* （2）1,2|3,4| 1,2不换位，3,4不换位，合并依然不换位 此时逆序对=0+0+0=0；
* （3）5,6,|7,0| 5,6不换位，7,0换位，合并换两次位 此时逆序对=0+1+2=3；
* （4）此时 1,2,3,4,0,5,6,7|换位4次，总逆序对=0+3+4=7；
#### 11.和为S的连续正数序列
##### 输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序。
```
public class Solution {
    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
         ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
	       ArrayList<Integer> list=new ArrayList<>();
	       ArrayList<Integer> listinput=new ArrayList<>();
	       if(sum==0||sum==1)
	           return result;
	        int jssum=0;
	        for(int i=1;i<=sum/2+2;i++){
	        	if(jssum<sum){
	        		jssum=jssum+i;
	        		list.add(i);
	        	}
	        	else if(jssum==sum){
	        		listinput=(ArrayList<Integer>) list.clone();
	        		result.add(listinput);
	        		i=list.get(0);
	        		list.removeAll(list);
	        		jssum=0;
	        	}
	        	else{
	        		i=list.get(0);
	        		list.clear();
	        		jssum=0;
	        	}
	        }
	        if(jssum==sum)
	        	result.add(list);
	        return result;
    }
}
```
* 对于这类问题，我们首先明白循环到sum>>1+2就可以了，因为我们知道一个数最多可以由（中值*2+1）取的。
#### 12.把数组排成最小的数
##### 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
```
public class Solution {
    public String PrintMinNumber(int [] numbers) {
        	int n;
	        String s="";
	        ArrayList<Integer> list=new ArrayList<Integer>();
	        n=numbers.length;	         
	        for(int i=0;i<n;i++){
	            list.add(numbers[i]);//将数组放入arrayList中
	        }
	        //实现了Comparator接口的compare方法，将集合元素按照compare方法的规则进行排序
	        Collections.sort(list,new Comparator<Integer>(){    
	            public int compare(Integer str1, Integer str2) {
	                // TODO Auto-generated method stub         
	                    String s1=str1+""+str2;
	                    String s2=str2+""+str1;	                     
	                    return s1.compareTo(s2);
	                }
	            });
	         
	        for(int j:list){
	            s+=j;
	        }
	        return s;
    }
}
```
* 使用Collections.sort做对比；举例：1,32,321；如果1和32组合最小值为132,32和321组合最小值为32132，所以排序为1,321,32.->132132
#### 13.二进制中1的个数
##### 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
```
public class Solution {
     public int NumberOf1(int n) {
		 	 int count=0;
			 String s=Integer.toBinaryString(n);
			 for(int i=0;i<s.length();i++){
				 if(s.charAt(i)=='1')
					 count++;
			 }
		 	return count;
	 }
}
```
* 主要运用Integer.toBinaryString（）方法转化为二进制；
#### 14.调整数组顺序使奇数位于偶数前面
##### 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
```
public class Solution {
    public void reOrderArray(int [] array) {
       ArrayList<Integer> arrayList1=new ArrayList<Integer>();
       ArrayList<Integer> arrayList2=new ArrayList<Integer>();
		for(int i=0;i<array.length;i++){
        	if(array[i]%2!=0)
        		arrayList1.add(array[i]);
        	else
        		arrayList2.add(array[i]);
        	}
		for(int j=0;j<arrayList1.size();j++){
			array[j]=arrayList1.get(j);
		}
		for(int m=0;m<arrayList2.size();m++){
			array[m+arrayList1.size()]=arrayList2.get(m);
		}  
    }
}
```
* 把偶数和奇数分别存在ArrayList中即可解决；
#### 15.链表中倒数第k个结点
##### 输入一个链表，输出该链表中倒数第k个结点。
```
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
import java.util.Stack;
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        Stack<ListNode> stack=new Stack<ListNode>();
        if(head==null||k==0)
            return null;
		 while(head!=null){
			 stack.push(head);
			 head=head.next;
		 }
		 for(int i=0;i<k-1;i++){
			 stack.pop();
		 }
         if(!stack.isEmpty())
		 return stack.pop();
		 else
		 return null;
    }
}
```
* 遍历链表到栈中，取出栈中第K个节点即可；
#### 16.反转链表
##### 输入一个链表，反转链表后，输出链表的所有元素。
```
public class Solution {
    public ListNode ReverseList(ListNode head) {
             if(head==null)
	            return null;
	        //head为当前节点，如果当前节点为空的话，那就什么也不做，直接返回null；
	        ListNode pre = null;
	        ListNode next = null;
	        //当前节点是head，pre为当前节点的前一节点，next为当前节点的下一节点
	        //需要pre和next的目的是让当前节点从pre->head->next1->next2变成pre<-head next1->next2
	        //即pre让节点可以反转所指方向，但反转之后如果不用next节点保存next1节点的话，此单链表就此断开了
	        //所以需要用到pre和next两个节点
	        while(head!=null){
	            //做循环，如果当前节点不为空的话，始终执行此循环，此循环的目的就是让当前节点从指向next到指向pre
	            //如此就可以做到反转链表的效果
	            //先用next保存head的下一个节点的信息，保证单链表不会因为失去head节点的原next节点而就此断裂
	            next = head.next;
	            //保存完next，就可以让head从指向next变成指向pre了，代码如下
	            head.next = pre;
	            //head指向pre后，就继续依次反转下一个节点
	            //让pre，head，next依次向后移动一个节点，继续下一次的指针反转
	            pre = head;
	            head = next;
	        }
	        //如果head为null的时候，pre就为最后一个节点了，但是链表已经反转完毕，pre就是反转后链表的第一个节点
	        //直接输出pre就是我们想要得到的反转后的链表
	        return pre;
    }
}
```
* 详见注解
#### 17.平衡二叉树
##### 输入一棵二叉树，判断该二叉树是否是平衡二叉树
```
package com.offer;

import com.entity.TreeNode;

public class 平衡二叉树 {

	public static void main(String[] args) {
		TreeNode tree1=new TreeNode(8);
		tree1.left=new TreeNode(6);
		tree1.right=new TreeNode(10);
		tree1.left.left=new TreeNode(5);
		tree1.left.right=new TreeNode(7);
		tree1.right.left=new TreeNode(9);
		tree1.right.right=new TreeNode(11);
		System.out.println(IsBalanced_Solution(tree1));
	}
	public static boolean IsBalanced_Solution(TreeNode root) {
        return getDepth(root) != -1;
    }
     
    private static int getDepth(TreeNode root) {
        if (root == null) return 0;
        int left = getDepth(root.left);
        if (left == -1) return -1;
        int right = getDepth(root.right);
        if (right == -1) return -1;
        return Math.abs(left - right) > 1 ? -1 : 1 + Math.max(left, right);
    }

}
```
* 从下往上进行判断，一旦出现左右子树>1，则说明整个树不平衡，减少计算量

